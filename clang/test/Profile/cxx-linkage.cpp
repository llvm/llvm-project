// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -emit-llvm -main-file-name cxx-linkage.cpp %s -o - -fprofile-instr-generate | FileCheck %s

// CHECK: @__profc__Z3foov = private global
// CHECK: @__profd__Z3foov = private global
// CHECK: @__profc__Z8foo_weakv = weak hidden global
// CHECK: @__profd__Z8foo_weakv = weak hidden global
// CHECK: @__profc_main = private global
// CHECK: @__profd_main = private global
// CHECK: @__profc__Z10foo_inlinev = linkonce_odr hidden global
// CHECK: @__profd__Z10foo_inlinev = linkonce_odr hidden global

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
