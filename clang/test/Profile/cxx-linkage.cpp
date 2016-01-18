// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -emit-llvm -main-file-name cxx-linkage.cpp %s -o - -fprofile-instr-generate | FileCheck %s

// CHECK: @__llvm_profile_name__Z3foov = private constant [7 x i8] c"_Z3foov"
// CHECK: @__llvm_profile_name__Z8foo_weakv = weak hidden constant [12 x i8] c"_Z8foo_weakv"
// CHECK: @__llvm_profile_name_main = private constant [4 x i8] c"main"
// CHECK: @__llvm_profile_name__Z10foo_inlinev = linkonce_odr hidden constant [15 x i8] c"_Z10foo_inlinev"

void foo(void) { }

void foo_weak(void) __attribute__((weak));
void foo_weak(void) { if (0){} if (0){} if (0){} if (0){} }

inline void foo_inline(void);
int main(void) {
  foo();
  foo_inline();
  foo_weak();
  return 0;
}

inline void foo_inline(void) { if (0){} if (0){} if (0){} if (0){} if (0){} if (0){}}
