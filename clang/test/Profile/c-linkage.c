// Check that the profiling names we create have the linkage we expect
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-linkage.c %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// CHECK: @__llvm_profile_name_foo = private constant [3 x i8] c"foo"
// CHECK: @__llvm_profile_name_foo_weak = weak hidden constant [8 x i8] c"foo_weak"
// CHECK: @__llvm_profile_name_main = private constant [4 x i8] c"main"
// CHECK: @__llvm_profile_name_c_linkage.c_foo_internal = private constant [24 x i8] c"c-linkage.c:foo_internal"

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
