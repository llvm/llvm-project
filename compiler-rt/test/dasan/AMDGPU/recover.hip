// RUN: %clang_hip_dasan -fsanitize-recover=daddress %s -o %t %hip_libs
// RUN: %run %t repeat 2>&1 | FileCheck %s --check-prefix=REPEAT \
// RUN:   --implicit-check-not=SUMMARY
// RUN: %run %t wave 2>&1 | FileCheck %s --check-prefix=WAVE \
// RUN:   --implicit-check-not=SUMMARY
// RUN: %run %t lane 2>&1 | FileCheck %s --check-prefix=LANE
// RUN: %if dasan-globals %{ %run %t global 2>&1 | FileCheck %s --check-prefix=GLOBAL %}
// RUN: %run %t shared 2>&1 | FileCheck %s --check-prefix=SHARED
// RUN: %clang_hip_dasan -fsanitize-recover=daddress -O1 %s -o %t %hip_libs
// RUN: %run %t repeat 2>&1 | FileCheck %s --check-prefix=REPEAT \
// RUN:   --implicit-check-not=SUMMARY
// RUN: %run %t wave 2>&1 | FileCheck %s --check-prefix=WAVE \
// RUN:   --implicit-check-not=SUMMARY
// RUN: %run %t lane 2>&1 | FileCheck %s --check-prefix=LANE
// RUN: %if dasan-globals %{ %run %t global 2>&1 | FileCheck %s --check-prefix=GLOBAL %}
// RUN: %run %t shared 2>&1 | FileCheck %s --check-prefix=SHARED

// A repeat of one program counter within a dispatch is reported once. The
// stores still land, so the memory past a fresh allocation must be mapped.
__global__ void walk(int *P, int N) {
#pragma clang loop unroll(disable)
  for (int I = 0; I < N; ++I)
    P[64 + I] = I;
}

__global__ void every_lane(int *P) { P[64 + threadIdxX()] = 1; }

__global__ void one_lane(int *P) {
  if (blockIdxX() == 2 && threadIdxX() == 33)
    P[64] = 1;
}

__device__ int Table[16];

__global__ void write_global(int *Out, int I) {
  Table[I] = 1;
  *Out = Table[0] + 1;
}

// Without LDS redzones the recovered write lands in the neighbouring object.
__global__ void write_shared(int *Out, int I) {
  __shared__ int Buf[16];
  __shared__ int Other[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  Other[T] = T;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0] + Other[0] + 1;
}

int main(int argc, char **argv) {
  int *P = nullptr, *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));

  int Want = 0;
  switch (argv[1][0]) {
  case 'r':
    walk<<<1, 1>>>(P, 200);
    break;
  case 'w':
    every_lane<<<1, 32>>>(P);
    break;
  case 'l':
    one_lane<<<4, 64>>>(P);
    break;
  case 'g':
    write_global<<<1, 16>>>(Out, 16);
    Want = 1;
    break;
  case 's':
    write_shared<<<1, 16>>>(Out, 16);
    Want = 66;
    break;
  }
  CHECK_HIP(hipDeviceSynchronize());

  int Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  return Want && Got != Want;
}

// REPEAT: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// REPEAT: in walk{{.*}}recover.hip
// REPEAT: is located 0 bytes after 256-byte region
// REPEAT: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// WAVE: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// WAVE: WRITE of size 4 at {{0x[0-9a-f]+}} thread ({{[0-9]+}},0,0) of block (0,0,0)
// WAVE: in every_lane{{.*}}recover.hip
// WAVE: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow

// LANE: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// LANE: WRITE of size 4 at {{0x[0-9a-f]+}} thread (33,0,0) of block (2,0,0)
// LANE: in one_lane{{.*}}recover.hip
// LANE: is located 0 bytes after 256-byte region

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in write_global{{.*}}recover.hip
// GLOBAL: is located 0 bytes after global variable 'Table' {{.*}} of size 64

// SHARED: ERROR: DeviceAddressSanitizer: shared-buffer-overflow
// SHARED: in write_shared{{.*}}recover.hip
// SHARED: is located 0 bytes after 64-byte region
