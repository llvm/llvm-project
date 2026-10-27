// A CUDA runtime stand-in for the tests in this directory. It logs the
// fatbinaries and kernels that the JIT'd host code registers and the kernels
// it launches. Nothing runs on a device, and device memory is host memory.
// The tests build it with -nostdlib, and the C library functions it calls
// resolve in the clang-repl process.

typedef __SIZE_TYPE__ size_t;

int printf(const char *, ...);
void *calloc(size_t, size_t);
void *memcpy(void *, const void *, size_t);
char *strstr(const char *, const char *);
size_t strcspn(const char *, const char *);

struct dim3 {
  unsigned x, y, z;
};

struct FatbinWrapper {
  int Magic;
  int Version;
  const char *Fatbin;
  void *Unused;
};

static const void *Kernels[16];
static const char *KernelNames[16];
static unsigned NumKernels;
static struct dim3 Grid, Block;
static void *Handle;

static void printDirective(const char *PTX, const char *Name) {
  const char *Directive = strstr(PTX, Name);
  if (Directive)
    printf(" %.*s", (int)strcspn(Directive, "\n"), Directive);
}

// The fatbinary from clang-repl holds one PTX entry: the fatbinary header
// (its size at offset 6), the entry header (its size at offset 4) and the
// NUL-terminated PTX.
void **__cudaRegisterFatBinary(struct FatbinWrapper *Wrapper) {
  const char *Fatbin = Wrapper->Fatbin;
  unsigned short HeaderSize = *(const unsigned short *)(Fatbin + 6);
  unsigned EntryHeaderSize = *(const unsigned *)(Fatbin + HeaderSize + 4);
  const char *PTX = Fatbin + HeaderSize + EntryHeaderSize;
  printf("cudart: fatbin");
  printDirective(PTX, ".version");
  printDirective(PTX, ".target");
  printf("\n");
  return &Handle;
}

void __cudaUnregisterFatBinary(void **FatbinHandle) {}

void __cudaRegisterFunction(void **FatbinHandle, const void *Kernel,
                            char *DeviceFun, const char *DeviceName,
                            int ThreadLimit, void *Tid, void *Bid,
                            void *BlockDim, void *GridDim, int *WarpSize) {
  if (NumKernels < 16) {
    Kernels[NumKernels] = Kernel;
    KernelNames[NumKernels++] = DeviceName;
  }
  printf("cudart: register %s\n", DeviceName);
}

int cudaConfigureCall(struct dim3 GridDim, struct dim3 BlockDim,
                      size_t SharedMem, void *Stream) {
  Grid = GridDim;
  Block = BlockDim;
  return 0;
}

int cudaSetupArgument(const void *Arg, size_t Size, size_t Offset) { return 0; }

int cudaLaunch(const void *Kernel) {
  const char *Name = "<unregistered>";
  for (unsigned I = 0; I < NumKernels; ++I)
    if (Kernels[I] == Kernel)
      Name = KernelNames[I];
  printf("cudart: launch %s grid (%u,%u,%u) block (%u,%u,%u)\n", Name, Grid.x,
         Grid.y, Grid.z, Block.x, Block.y, Block.z);
  return 0;
}

int cudaMalloc(void **Ptr, size_t Size) {
  *Ptr = calloc(1, Size);
  return *Ptr ? 0 : 2; // cudaErrorMemoryAllocation
}

int cudaMemcpy(void *Dst, const void *Src, size_t Count, int Kind) {
  memcpy(Dst, Src, Count);
  return 0;
}

int cudaGetLastError(void) { return 0; }
