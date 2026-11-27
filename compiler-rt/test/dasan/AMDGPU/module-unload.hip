// REQUIRES: dasan-globals
// RUN: %clang_hip_dasan --offload-device-only %s -o %t.co
// RUN: %clang_dasan -x c++ -DHOST %s -o %t %hip_libs
// RUN: %run %t %t.co 2>&1 | FileCheck %s

// Loading the same image again after unloading it must reuse the span it held.

#ifdef HOST

extern "C" {
typedef struct ihipModule_t *hipModule_t;
typedef struct ihipModuleSymbol_t *hipFunction_t;
typedef struct ihipStream_t *hipStream_t;
int hipModuleLoad(hipModule_t *Module, const char *File);
int hipModuleUnload(hipModule_t Module);
int hipModuleGetFunction(hipFunction_t *Function, hipModule_t Module,
                         const char *Name);
int hipModuleLaunchKernel(hipFunction_t Function, unsigned GridX,
                          unsigned GridY, unsigned GridZ, unsigned BlockX,
                          unsigned BlockY, unsigned BlockZ, unsigned SharedMem,
                          hipStream_t Stream, void **Params, void **Extra);
int hipDeviceSynchronize(void);
}

int main(int argc, char **argv) {
  for (int N = 0; N < 2; ++N) {
    hipModule_t Module = nullptr;
    hipFunction_t Fn = nullptr;
    if (hipModuleLoad(&Module, argv[1]) ||
        hipModuleGetFunction(&Fn, Module, "touch") ||
        hipModuleLaunchKernel(Fn, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr) ||
        hipDeviceSynchronize() || hipModuleUnload(Module))
      return 2;
  }
  return 0;
}

#else
extern "C" __global__ void touch() {}
#endif

// CHECK: placed {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ADDR:0x[0-9a-f]+]]
// CHECK: released {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ADDR]]
// CHECK: placed {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ADDR]]
