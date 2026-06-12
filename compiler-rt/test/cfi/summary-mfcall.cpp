// RUN: %clangxx_cfi_diag -g -o %t %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t nvmfcall 2>&1 | FileCheck --check-prefix=NVMFCALL %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t vmfcall 2>&1 | FileCheck --check-prefix=VMFCALL %s

// UNSUPPORTED: target={{.*windows-msvc.*}}
// REQUIRES: cxxabi

#include "utils.h"
#include <stdio.h>
#include <string.h>

struct S {
  void f();         // non-virtual
  virtual void g(); // virtual
};
void S::f() {}
void S::g() {}

struct T {
  void f();
  virtual void g();
};
void T::f() {}
void T::g() {}

typedef void (S::*S_void)();
typedef int (S::*S_int)();

template <typename To, typename From> To bitcast(From f) {
  To t;
  memcpy(&t, &f, sizeof(f));
  return t;
}

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;

  create_derivers<T>();

  if (strcmp(argv[1], "nvmfcall") == 0) {
    S s;
    break_optimization(&s);
    // NVMFCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-mfcall {{.*}}summary-mfcall.cpp:[[@LINE+1]]
    (s.*bitcast<S_void>(&T::f))();
  } else if (strcmp(argv[1], "vmfcall") == 0) {
    S s;
    break_optimization(&s);
    // VMFCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-mfcall {{.*}}summary-mfcall.cpp:[[@LINE+1]]
    (s.*bitcast<S_int>(&S::g))();
  }

  return 0;
}
