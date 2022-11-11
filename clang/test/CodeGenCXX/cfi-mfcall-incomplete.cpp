// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-mfcall -fsanitize-trap=cfi-mfcall -fvisibility=hidden -emit-llvm -o - %s | FileCheck %s

struct S;

void f(S *s, void (S::*p)()) {
  // CHECK-NOT: llvm.type.test
  // CHECK: llvm.type.test{{.*}}!"_ZTSM1SFvvE.virtual"
  // CHECK-NOT: llvm.type.test
  (s->*p)();
}

// CHECK: declare i1 @llvm.type.test
