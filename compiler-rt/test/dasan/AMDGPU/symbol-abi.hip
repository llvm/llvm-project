// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %run %t | FileCheck %s
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %run %t | FileCheck %s

// A placed global's symbol describes the object alone, not its padded storage,
// so the host's view of its size and address is unchanged.
__device__ int Table[16];

__global__ void sum(int *Out) {
  int S = 0;
  for (int I = 0; I < 16; ++I)
    S += Table[I];
  *Out = S;
}

int main() {
  unsigned long Size = 0;
  CHECK_HIP(hipGetSymbolSize(&Size, (const void *)&Table));
  printf("size %lu\n", Size);

  void *Addr = nullptr;
  CHECK_HIP(hipGetSymbolAddress(&Addr, (const void *)&Table));
  printf("aligned %d\n", (int)((unsigned long)Addr % 64 == 0));

  int Host[16];
  for (int I = 0; I < 16; ++I)
    Host[I] = I + 1;

  CHECK_HIP(hipMemcpyToSymbol((const void *)&Table, Host, sizeof(Host)));
  CHECK_HIP(hipMemcpyToSymbol((const void *)&Table, Host, sizeof(Host) / 2,
                              sizeof(Host) / 2));
  printf("refused %d\n",
         (int)(hipMemcpyToSymbol((const void *)&Table, Host, Size + 1) != 0));
  hipGetLastError();

  int Back[8] = {0};
  CHECK_HIP(hipMemcpyFromSymbol(Back, (const void *)&Table, sizeof(Back),
                                sizeof(Back)));
  printf("offset %d\n", (int)(Back[0] == 1 && Back[7] == 8));

  int *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(int)));
  sum<<<1, 1>>>(Out);
  CHECK_HIP(hipDeviceSynchronize());

  int Got = 0;
  CHECK_HIP(hipMemcpy(&Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  printf("sum %d\n", Got);
  return 0;
}

// CHECK: size 64
// CHECK: aligned 1
// CHECK: refused 1
// CHECK: offset 1
// CHECK: sum 72
