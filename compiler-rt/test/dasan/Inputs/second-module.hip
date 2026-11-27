__device__ int Other[16];

__global__ void write_other(int I) { Other[I] = 1; }

__global__ void write_other_shared(int *Out, int I) {
  __shared__ int Buf[16];
  unsigned T = threadIdxX();
  Buf[T] = T;
  syncThreads();
  if (T == 0)
    Buf[I] = 0x41;
  syncThreads();
  *Out = Buf[0];
}
