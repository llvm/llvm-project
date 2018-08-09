// Check that cross-DSO diagnostics print the names of both modules

// RUN: %clangxx_cfi_diag -g -DSHARED_LIB -fPIC -shared -o %dynamiclib %s %ld_flags_rpath_so
// RUN: %clangxx_cfi_diag -g -o %t_exe_suffix %s %ld_flags_rpath_exe
// RUN: %t_exe_suffix 2>&1 | FileCheck -DDSONAME=%xdynamiclib_namespec %s

// UNSUPPORTED: windows-msvc
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

int main() {
  void* (*fp)(void) =
      reinterpret_cast<void*(*)(void)>(dlsym(RTLD_DEFAULT, "dso_symbol"));
  if (!fp) {
    perror("failed to resolve dso_symbol");
    return 1;
  }

  // CHECK: runtime error: control flow integrity check for type 'void *()' failed during indirect function call
  // CHECK: dso_symbol defined here
  // CHECK: check failed in {{.*}}_exe_suffix, destination function located in {{.*}}[[DSONAME]]
  void *S = fp(); // trigger cfi-icall failure

  // CHECK: runtime error: control flow integrity check for type 'S1' failed during cast to unrelated type
  // CHECK: invalid vtable
  // CHECK: check failed in {{.*}}_exe_suffix, vtable located in {{.*}}[[DSONAME]]
  S1 *Scast = reinterpret_cast<S1*>(S); // trigger cfi-unrelated-cast failure

  return 0;
}

#endif // SHARED_LIB
