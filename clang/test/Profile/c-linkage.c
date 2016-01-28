// Check that the profiling counters and data we create have the linkage we expect
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-linkage.c %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// CHECK: @__profc_foo = private global
// CHECK: @__profd_foo = private global
// CHECK: @__profc_foo_weak = weak hidden global
// CHECK: @__profd_foo_weak = weak hidden global
// CHECK: @__profc_main = private global
// CHECK: @__profd_main = private global
// CHECK: @__profc_c_linkage.c_foo_internal = private global
// CHECK: @__profd_c_linkage.c_foo_internal = private global

void foo(void) { }

void foo_weak(void) __attribute__((weak));
void foo_weak(void) { if (0){} if (0){} if (0){} if (0){} }

static void foo_internal(void);
int main(void) {
  foo();
  foo_internal();
  foo_weak();
  return 0;
}

static void foo_internal(void) { if (0){} if (0){} }
