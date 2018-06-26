// Check that cross-DSO diagnostics print the names of both modules

// RUN: %clangxx_cfi_diag -g -DSHARED_LIB -fPIC -shared -o %t_dso_suffix %s
// RUN: %clangxx_cfi_diag -g -o %t_exe_suffix %s -ldl
// RUN: %t_exe_suffix %t_dso_suffix 2>&1 | FileCheck %s

// UNSUPPORTED: win32
// REQUIRES: cxxabi

#include <dlfcn.h>
#include <stdio.h>

struct S1 {
  virtual void f1();
};

#ifdef SHARED_LIB

void S1::f1() {}

__attribute__((visibility("default"))) extern "C"
void* dso_symbol() { return new S1(); }

#else

int main(int argc, char *argv[]) {
  void *handle = dlopen(argv[1], RTLD_NOW);

  // CHECK: runtime error: control flow integrity check for type 'void *()' failed during indirect function call
  // CHECK: dso_symbol defined here
  // CHECK: check failed in {{.*}}_exe_suffix, destination function located in {{.*}}_dso_suffix
  void* (*fp)(void) =
      reinterpret_cast<void*(*)(void)>(dlsym(handle, "dso_symbol"));
  void *S = fp(); // trigger cfi-icall failure

  // CHECK: runtime error: control flow integrity check for type 'S1' failed during cast to unrelated type
  // CHECK: invalid vtable
  // CHECK: check failed in {{.*}}_exe_suffix, vtable located in {{.*}}_dso_suffix
  S1 *Scast = reinterpret_cast<S1*>(S); // trigger cfi-unrelated-cast failure
}

#endif // SHARED_LIB
