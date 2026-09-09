// RUN: %clang_cc1 -fsyntax-only -std=c17 -fexperimental-lifetime-safety-c \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -std=c23 -fexperimental-lifetime-safety-c \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=CHECK-C23
// RUN: %clang_cc1 -fsyntax-only -std=c17 -fexperimental-lifetime-safety-c \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   '-DLIFETIMEBOUND_MACRO=__attribute__((lifetimebound))' \
// RUN:   -lifetime-safety-lifetimebound-macro=LIFETIMEBOUND_MACRO \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACRO
// RUN: cp %s %t.c
// RUN: %clang_cc1 -std=c17 -fexperimental-lifetime-safety-c \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling -fixit %t.c
// RUN: %clang_cc1 -fsyntax-only -std=c17 -fexperimental-lifetime-safety-c \
// RUN:   -Werror=lifetime-safety-suggestions -Wno-dangling %t.c

int *return_pointer(int *p) {
  // CHECK: :[[@LINE-1]]:27: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:27-[[@LINE-2]]:27}:" __attribute__((lifetimebound))"
  // CHECK-C23: :[[@LINE-3]]:27: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:27-[[@LINE-4]]:27}:" {{\[\[}}clang::lifetimebound]]"
  // CHECK-MACRO: :[[@LINE-5]]:27: warning: parameter in intra-TU function should be marked
  // CHECK-MACRO: fix-it:"{{.*}}":{[[@LINE-6]]:27-[[@LINE-6]]:27}:" LIFETIMEBOUND_MACRO"
  return p;
}

int *return_multi(int *a, int cond, int *b) {
  // CHECK-DAG: :[[@LINE-1]]:25: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:" __attribute__((lifetimebound))"
  // CHECK-DAG: :[[@LINE-3]]:43: warning: parameter in intra-TU function should be marked
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-4]]:43-[[@LINE-4]]:43}:" __attribute__((lifetimebound))"
  if (cond)
    return a;
  return b;
}

int *return_partial(int *a __attribute__((lifetimebound)), int cond, int *b) {
  // CHECK: :[[@LINE-1]]:76: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:76-[[@LINE-2]]:76}:" __attribute__((lifetimebound))"
  // CHECK-C23: :[[@LINE-3]]:76: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:76-[[@LINE-4]]:76}:" {{\[\[}}clang::lifetimebound]]"
  if (cond)
    return a;
  return b;
}

#define GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

int *unnamed_pointer(int *);
// CHECK: :[[@LINE-1]]:22: warning: parameter in intra-TU function should be marked
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:22-[[@LINE-2]]:22}:"GNU_LIFETIMEBOUND_MACRO "
// CHECK-C23: :[[@LINE-3]]:22: warning: parameter in intra-TU function should be marked
// CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:22}:"GNU_LIFETIMEBOUND_MACRO "
int *unnamed_pointer(int *p) {
  return p;
}

int *return_pointer_with_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:42: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:42-[[@LINE-2]]:42}:" GNU_LIFETIMEBOUND_MACRO"
  // CHECK-C23: :[[@LINE-3]]:42: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:42-[[@LINE-4]]:42}:" GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#define FIRST_GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))
#define SECOND_GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

int *return_pointer_with_latest_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:49: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:49-[[@LINE-2]]:49}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  // CHECK-C23: :[[@LINE-3]]:49: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:49-[[@LINE-4]]:49}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#define REDEFINED_GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

int *return_pointer_with_redefined_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:52: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:52-[[@LINE-2]]:52}:" REDEFINED_GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#undef REDEFINED_GNU_LIFETIMEBOUND_MACRO
#define REDEFINED_GNU_LIFETIMEBOUND_MACRO __attribute__((unused))

int *return_pointer_after_redefined_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:53: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:53-[[@LINE-2]]:53}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#define UNDEFINED_GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

int *return_pointer_with_undefined_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:52: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:52-[[@LINE-2]]:52}:" UNDEFINED_GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#undef UNDEFINED_GNU_LIFETIMEBOUND_MACRO

int *return_pointer_after_undefined_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:53: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:53-[[@LINE-2]]:53}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  return p;
}

#define CLANG_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

int *return_pointer_with_clang_macro(int *p) {
  // CHECK: :[[@LINE-1]]:44: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:44-[[@LINE-2]]:44}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  // CHECK-C23: :[[@LINE-3]]:44: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:44-[[@LINE-4]]:44}:" CLANG_LIFETIMEBOUND_MACRO"
  return p;
}

#define FIRST_CLANG_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]
#define SECOND_CLANG_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

int *return_pointer_with_latest_clang_macro(int *p) {
  // CHECK: :[[@LINE-1]]:51: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:51-[[@LINE-2]]:51}:" SECOND_GNU_LIFETIMEBOUND_MACRO"
  // CHECK-C23: :[[@LINE-3]]:51: warning: parameter in intra-TU function should be marked
  // CHECK-C23: fix-it:"{{.*}}":{[[@LINE-4]]:51-[[@LINE-4]]:51}:" SECOND_CLANG_LIFETIMEBOUND_MACRO"
  // CHECK-MACRO: :[[@LINE-5]]:51: warning: parameter in intra-TU function should be marked
  // CHECK-MACRO: fix-it:"{{.*}}":{[[@LINE-6]]:51-[[@LINE-6]]:51}:" LIFETIMEBOUND_MACRO"
  return p;
}
