// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %run %t
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %run %t

// Every kind of covered memory, touched only in bounds.
__device__ int Global[16];

__global__ void touch_all(int *Heap, int N) {
  __shared__ int Shared[16];
  extern __shared__ int Dyn[];
  unsigned T = threadIdxX();

  if (T < 16) {
    Shared[T] = T;
    Global[T] = T;
  }
  Dyn[T] = T;
  syncThreads();

  for (int I = T; I < N; I += 16)
    Heap[I] = Shared[I % 16] + Global[I % 16] + Dyn[T];
}

int main() {
  const int N = 256;
  int *Heap = nullptr;
  CHECK_HIP(hipMalloc((void **)&Heap, N * sizeof(int)));
  touch_all<<<1, 16, 16 * sizeof(int)>>>(Heap, N);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}
