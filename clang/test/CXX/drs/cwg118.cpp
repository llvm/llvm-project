// RUN: %clang_cc1 -triple x86_64-linux -std=c++98 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++11 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++14 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++17 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++20 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++23 %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "
// RUN: %clang_cc1 -triple x86_64-linux -std=c++2c %s -pedantic-errors -emit-llvm -o - | FileCheck %s --implicit-check-not " call "


// cwg118: yes

struct S {
  virtual void f();
};
void (S::*pmf)();

// CHECK-LABEL: define {{.*}} @_Z1g
void g(S *sp) {
  // CHECK: call void %
  sp->f();        // 1: polymorphic
  // CHECK: call void @
  sp->S::f();     // 2: non-polymorphic
  // CHECK: call void @
  (sp->S::f)();   // 3: non-polymorphic
  // CHECK: call void %
  (sp->*pmf)();   // 4: polymorphic
  // CHECK: call void %
  (sp->*&S::f)(); // 5: polymorphic
}

