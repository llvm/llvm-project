// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %run %t
// RUN: not %run %t trip 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %run %t
// RUN: not %run %t trip 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %run %t storm 2>&1 | FileCheck %s --check-prefix=STORM
// RUN: %clang_hip_dasan -fsanitize-recover=daddress -O1 %s -o %t %hip_libs
// RUN: %run %t storm 2>&1 | FileCheck %s --check-prefix=STORM

__device__ int Global[1024];
__constant__ int Konst[64] = {1};

__global__ void sweep(char *P, unsigned long N, unsigned long Stride,
                      unsigned long long *Sum) {
  unsigned long T = threadIdxX() + (unsigned long)blockIdxX() * 256;
  unsigned long long Acc = 0;
  for (unsigned long I = T; I < N; I += Stride) {
    P[I] = (char)(I + 1);
    Acc += (unsigned char)P[I];
  }
  __atomic_fetch_add(Sum, Acc, __ATOMIC_RELAXED);
}

__global__ void shuffle(char *Dst, const char *Src, unsigned long N) {
  if (threadIdxX() == 0) {
    __builtin_memset(Dst, 0x41, N);
    __builtin_memcpy(Dst, Src, N);
  }
}

__global__ void locals(unsigned long long *Sum) {
  __shared__ int Buf[256];
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();

  Buf[T] = T;
  Dyn[T] = T;
  Global[T + 256 * blockIdxX()] = T;
  syncThreads();

  unsigned long long Acc = Buf[(T + 1) % 256] + Dyn[(T + 1) % 256] +
                           Global[T + 256 * blockIdxX()] + Konst[T % 64];
  __atomic_fetch_add(Sum, Acc, __ATOMIC_RELAXED);
}

__global__ void around(int *Q, int Half) {
#pragma clang loop unroll(disable)
  for (int I = -Half; I < Half; ++I)
    Q[I] = I;
}

__global__ void poke(char *P, long I) { P[I] = 1; }

static const unsigned long Sizes[] = {16,   64,   252,   256,
                                      1000, 4096, 33000, 100000};
static const int NumSizes = sizeof(Sizes) / sizeof(Sizes[0]);

int main(int argc, char **argv) {
  const char Mode = argc > 1 ? argv[1][0] : 0;

  unsigned long long *Sum = nullptr;
  CHECK_HIP(hipMalloc((void **)&Sum, sizeof(*Sum)));
  CHECK_HIP(hipMemset(Sum, 0, sizeof(*Sum)));

  char *Live[NumSizes] = {nullptr};
  unsigned long long Want = 0;
  const int Rounds = Mode ? 2 : 4;

  for (int R = 0; R < Rounds; ++R) {
    for (int S = 0; S < NumSizes; ++S) {
      const unsigned long N = Sizes[S];
      char *P = nullptr;
      CHECK_HIP(hipMalloc((void **)&P, N));
      if ((unsigned long)P % 256)
        return 1;

      const unsigned Blocks = N > 4096 ? 8 : 2;
      sweep<<<Blocks, 256>>>(P, N, 256UL * Blocks, Sum);
      for (unsigned long I = 0; I < N; ++I)
        Want += (unsigned char)(I + 1);

      if (Live[S]) {
        shuffle<<<1, 1>>>(P, Live[S], N);
        CHECK_HIP(hipFree(Live[S]));
        Live[S] = nullptr;
      }

      if (N >= 256)
        around<<<1, 1>>>((int *)P + N / 8, (int)(N / 8));

      // Half are kept for the next round so every round both frees and holds.
      if (S & 1)
        Live[S] = P;
      else
        CHECK_HIP(hipFree(P));
    }

    locals<<<4, 256, 256 * sizeof(int)>>>(Sum);
    for (int T = 0; T < 4 * 256; ++T)
      Want += (T % 256) + (T % 256) + (T % 256) + (T % 64 ? 0 : 1);
  }
  CHECK_HIP(hipDeviceSynchronize());

  unsigned long long Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Sum, sizeof(Got), hipMemcpyDeviceToHost));
  if (Got != Want)
    return 1;

  if (Mode == 't')
    poke<<<1, 1>>>(Live[7], (long)Sizes[7]);
  // The launches share a program counter, so the site is reported once.
  if (Mode == 's')
    for (int R = 0; R < 4; ++R)
      poke<<<8, 64>>>(Live[5], (long)Sizes[5]);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in poke{{.*}}stress.hip
// CHECK: [[ADDR]] is located 0 bytes after 100000-byte region

// STORM: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// STORM: is located 0 bytes after 4096-byte region
