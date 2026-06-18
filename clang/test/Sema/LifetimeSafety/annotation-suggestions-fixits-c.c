// RUN: %clang_cc1 -fsyntax-only -std=c17 \
// RUN:   -fexperimental-lifetime-safety-c -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -std=c17 \
// RUN:   -fexperimental-lifetime-safety-c -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling \
// RUN:   -lifetime-safety-lifetimebound-macro=CONFIGURED_LIFETIMEBOUND_MACRO \
// RUN:   -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=CHECK-CONFIG
// RUN: cp %s %t.c
// RUN: %clang_cc1 -std=c17 \
// RUN:   -fexperimental-lifetime-safety-c -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Wno-dangling -fixit %t.c
// RUN: %clang_cc1 -fsyntax-only -std=c17 \
// RUN:   -fexperimental-lifetime-safety-c -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Werror=lifetime-safety-suggestions -Wno-dangling %t.c

int *return_ptr(int *p) {
  // CHECK: :[[@LINE-1]]:17: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:23-[[@LINE-2]]:23}:" __attribute__((lifetimebound))"
  return p;
}

#define CLANG_LIFETIMEBOUND_MACRO [[clang::lifetimebound]]

int *return_ptr_with_clang_macro(int *p) {
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:{{[0-9]+}}-[[@LINE-2]]:{{[0-9]+}}}:" __attribute__((lifetimebound))"
  return p;
}

#define CONFIGURED_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))
#define GNU_LIFETIMEBOUND_MACRO __attribute__((lifetimebound))

int *return_ptr_with_gnu_macro(int *p) {
  // CHECK: :[[@LINE-1]]:{{[0-9]+}}: warning: parameter in intra-TU function should be marked
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:{{[0-9]+}}-[[@LINE-2]]:{{[0-9]+}}}:" GNU_LIFETIMEBOUND_MACRO"
  // CHECK-CONFIG: :[[@LINE-3]]:{{[0-9]+}}: warning: parameter in intra-TU function should be marked
  // CHECK-CONFIG: fix-it:"{{.*}}":{[[@LINE-4]]:{{[0-9]+}}-[[@LINE-4]]:{{[0-9]+}}}:" CONFIGURED_LIFETIMEBOUND_MACRO"
  return p;
}
