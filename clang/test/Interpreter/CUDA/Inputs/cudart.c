// Stub CUDA runtime for the tests in this directory. Testing purposes only.

typedef __SIZE_TYPE__ size_t;

void *calloc(size_t, size_t);
void *memcpy(void *, const void *, size_t);

struct dim3 {
  unsigned x, y, z;
};

static void *Handle;

void **__cudaRegisterFatBinary(void *Wrapper) { return &Handle; }

void __cudaUnregisterFatBinary(void **FatbinHandle) {}

void __cudaRegisterFunction(void **FatbinHandle, const void *Kernel,
                            char *DeviceFun, const char *DeviceName,
                            int ThreadLimit, void *Tid, void *Bid,
                            void *BlockDim, void *GridDim, int *WarpSize) {}

int cudaConfigureCall(struct dim3 GridDim, struct dim3 BlockDim,
                      size_t SharedMem, void *Stream) {
  return 0;
}

int cudaSetupArgument(const void *Arg, size_t Size, size_t Offset) { return 0; }

int cudaLaunch(const void *Kernel) { return 0; }

int cudaMalloc(void **Ptr, size_t Size) {
  *Ptr = calloc(1, Size);
  return 0;
}

int cudaMemcpy(void *Dst, const void *Src, size_t Count, int Kind) {
  memcpy(Dst, Src, Count);
  return 0;
}

int cudaGetLastError(void) { return 0; }
