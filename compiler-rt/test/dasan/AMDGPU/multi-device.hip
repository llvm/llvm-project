// RUN: %clang_hip_dasan %s -o %t %hip_libs -pthread
// RUN: %run %t
// RUN: not %run %t heap 0 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t heap 1 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: %if dasan-globals %{ not %run %t global 0 2>&1 | FileCheck %s --check-prefix=GLOBAL %}
// RUN: %if dasan-globals %{ not %run %t global 1 2>&1 | FileCheck %s --check-prefix=GLOBAL %}
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs -pthread
// RUN: %run %t
// RUN: not %run %t heap 0 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t heap 1 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: %if dasan-globals %{ not %run %t global 0 2>&1 | FileCheck %s --check-prefix=GLOBAL %}
// RUN: %if dasan-globals %{ not %run %t global 1 2>&1 | FileCheck %s --check-prefix=GLOBAL %}

// REQUIRES: multi-device

#ifndef __HIP_DEVICE_COMPILE__
#  include <pthread.h>
#endif

__device__ int Table[16];

__global__ void fill(int *P, int N, int Tag) {
  unsigned T = threadIdxX();
  if ((int)T < N)
    P[T] = Tag + (int)T;
}

__global__ void poke_heap(int *P, int I) { P[I] = 1; }

__global__ void poke_global(int I) { Table[I] = 1; }

#ifndef __HIP_DEVICE_COMPILE__

static const int N = 64;

struct Job {
  int Device;
  int Tag;
  int *Live;
  int Rc;
};

static int *Place(int Device) {
  int *P = nullptr;
  if (hipSetDevice(Device) || hipMalloc((void **)&P, N * sizeof(int)))
    return nullptr;
  return P;
}

static int Run(int Device, int *P, int Tag) {
  if (hipSetDevice(Device))
    return 2;
  fill<<<1, N>>>(P, N, Tag);
  return hipDeviceSynchronize();
}

static int Release(int Device, int *P) {
  return hipSetDevice(Device) || hipFree(P);
}

static void *Work(void *Arg) {
  Job *J = (Job *)Arg;
  J->Live = Place(J->Device);
  J->Rc = J->Live ? Run(J->Device, J->Live, J->Tag) : 2;
  if (J->Live && Release(J->Device, J->Live))
    J->Rc = 2;
  return nullptr;
}

int main(int argc, char **argv) {
  int Devices = 0;
  CHECK_HIP(hipGetDeviceCount(&Devices));
  if (Devices < 2)
    return 2;

  if (argc > 2) {
    const int Dev = argv[2][0] - '0';
    if (argv[1][0] == 'h') {
      int *P = Place(Dev);
      if (!P)
        return 2;
      poke_heap<<<1, 1>>>(P, N);
    } else {
      CHECK_HIP(hipSetDevice(Dev));
      poke_global<<<1, 1>>>(16);
    }
    CHECK_HIP(hipDeviceSynchronize());
    return 0;
  }

  // Each device on its own.
  for (int D = 0; D < 2; ++D) {
    Job J = {D, D + 1, nullptr, 0};
    Work(&J);
    if (J.Rc)
      return 2;
  }

  // Both heaps live in the same size class at once.
  int *A = Place(0);
  int *B = Place(1);
  if (!A || !B || Run(0, A, 10) || Run(1, B, 20) || Release(0, A) ||
      Release(1, B))
    return 2;

  // One host thread per device, through the interceptors at once.
  Job J0 = {0, 30, nullptr, 0};
  Job J1 = {1, 40, nullptr, 0};
  pthread_t T0, T1;
  if (pthread_create(&T0, nullptr, Work, &J0) ||
      pthread_create(&T1, nullptr, Work, &J1))
    return 2;
  if (pthread_join(T0, nullptr) || pthread_join(T1, nullptr))
    return 2;
  return J0.Rc || J1.Rc ? 2 : 0;
}

#endif

// HEAP: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// HEAP: WRITE of size 4
// HEAP: in poke_heap{{.*}}multi-device.hip
// HEAP: is located 0 bytes after 256-byte region

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in poke_global{{.*}}multi-device.hip
// GLOBAL: is located 0 bytes after global variable 'Table' {{.*}} of size 64
