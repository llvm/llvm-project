// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -main-file-name call-continuations-hash.c -fprofile-instrument=clang -fcoverage-mapping -emit-llvm -o - %s | FileCheck %s --check-prefix=NOCC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -main-file-name call-continuations-hash.c -fprofile-instrument=clang -fcoverage-mapping -fcoverage-call-continuations -emit-llvm -o - %s | FileCheck %s --check-prefix=CC

void f(void);

int foo(void) {
  f();
  return 1;
}

// NOCC: @__profd_foo = private global {{.*}} { i64 {{[-0-9]+}}, i64 24, {{.*}}, i32 1,
// CC: @__profd_foo = private global {{.*}} { i64 {{[-0-9]+}}, i64 1569, {{.*}}, i32 2,
