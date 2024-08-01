// RUN: %clang_cc1 -triple x86_64-windows-msvc %s -emit-llvm -fms-extensions -fms-compatibility -fms-reference-binding -o - | FileCheck %s

struct A {};
struct B : A {};

void fAPickConstRef(A&) {}
void fAPickConstRef(const A&) {}

void fBPickConstRef(A&) {}
void fBPickConstRef(const A&) {}

void fAPickRef(A&) {}
void fAPickRef(const volatile A&) {}

void fAPickRef2(A&) {}
void fAPickRef2(const volatile A&) {}

void test() {
  fAPickConstRef(A());
  // CHECK: call {{.*}} @"?fAPickConstRef@@YAXAEBUA@@@Z"

  fBPickConstRef(B());
  // CHECK: call {{.*}} @"?fBPickConstRef@@YAXAEBUA@@@Z"

  fAPickRef(A());
  // CHECK: call {{.*}} @"?fAPickRef@@YAXAEAUA@@@Z"

  fAPickRef2(A());
  // CHECK: call {{.*}} @"?fAPickRef2@@YAXAEAUA@@@Z"
}
