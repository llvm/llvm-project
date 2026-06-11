// RUN: %clangxx_cfi_diag -g -o %t %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t vcall 2>&1 | FileCheck --check-prefix=VCALL %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t nvcall 2>&1 | FileCheck --check-prefix=NVCALL %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t ucast 2>&1 | FileCheck --check-prefix=UCAST %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t dcast 2>&1 | FileCheck --check-prefix=DCAST %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t icall 2>&1 | FileCheck --check-prefix=ICALL %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t nvmfcall 2>&1 | FileCheck --check-prefix=NVMFCALL %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 %run %t vmfcall 2>&1 | FileCheck --check-prefix=VMFCALL %s

// REQUIRES: cxxabi

#include "utils.h"
#include <stdio.h>
#include <string.h>

struct A {
  virtual void f();
};
void A::f() {}

struct B {
  virtual void f();
};
void B::f() {}

struct C : A {
  virtual void f();
};
void C::f() {}

struct D {
  void f(); // non-virtual
  virtual void g();
};
void D::f() {}
void D::g() {}

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

void g() {}

int main(int argc, char **argv) {
  if (argc < 2)
    return 0;

  create_derivers<B>();
  create_derivers<D>();
  create_derivers<T>();

  if (strcmp(argv[1], "vcall") == 0) {
    A *a = new A;
    break_optimization(a);
    // VCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-vcall {{.*}}summary.cpp:[[@LINE+1]]
    ((B *)a)->f();
  } else if (strcmp(argv[1], "nvcall") == 0) {
    A *a = new A;
    break_optimization(a);
    // NVCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-nvcall {{.*}}summary.cpp:[[@LINE+1]]
    ((D *)a)->f();
  } else if (strcmp(argv[1], "ucast") == 0) {
    A *a = new A;
    break_optimization(a);
    // UCAST: SUMMARY: UndefinedBehaviorSanitizer: cfi-unrelated-cast {{.*}}summary.cpp:[[@LINE+1]]
    B *b = (B *)a;
    break_optimization(b);
  } else if (strcmp(argv[1], "dcast") == 0) {
    A *a = new A;
    break_optimization(a);
    // DCAST: SUMMARY: UndefinedBehaviorSanitizer: cfi-derived-cast {{.*}}summary.cpp:[[@LINE+1]]
    C *c = (C *)a;
    break_optimization(c);
  } else if (strcmp(argv[1], "icall") == 0) {
    void (*fp)() = g;
    break_optimization(&fp);
    // ICALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-icall {{.*}}summary.cpp:[[@LINE+1]]
    ((void (*)(int))fp)(42);
  } else if (strcmp(argv[1], "nvmfcall") == 0) {
    S s;
    break_optimization(&s);
    // NVMFCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-mfcall {{.*}}summary.cpp:[[@LINE+1]]
    (s.*bitcast<S_void>(&T::f))();
  } else if (strcmp(argv[1], "vmfcall") == 0) {
    S s;
    break_optimization(&s);
    // VMFCALL: SUMMARY: UndefinedBehaviorSanitizer: cfi-mfcall {{.*}}summary.cpp:[[@LINE+1]]
    (s.*bitcast<S_int>(&S::g))();
  }

  return 0;
}
