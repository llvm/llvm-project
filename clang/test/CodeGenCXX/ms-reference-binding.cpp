// RUN: %clang_cc1 -triple x86_64-windows-msvc %s -emit-llvm -fms-extensions -fms-compatibility -fms-reference-binding -Wno-microsoft-reference-binding -o - | FileCheck %s

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

namespace NS {
  void fAPickConstRef(A&) {}
  void fAPickConstRef(const A&) {}

  void fBPickConstRef(A&) {}
  void fBPickConstRef(const A&) {}

  void fAPickRef(A&) {}
  void fAPickRef(const volatile A&) {}

  void fAPickRef2(A&) {}
  void fAPickRef2(const volatile A&) {}
}

struct S {
  void memberPickNonConst() {}
  void memberPickNonConst() const {}

  void memberPickConstRef() const & {}
  void memberPickConstRef() & {}

  static void fAPickConstRef(A&) {}
  static void fAPickConstRef(const A&) {}
};

void test() {
  fAPickConstRef(A());
  // CHECK: call {{.*}} @"?fAPickConstRef@@YAXAEBUA@@@Z"

  fBPickConstRef(B());
  // CHECK: call {{.*}} @"?fBPickConstRef@@YAXAEBUA@@@Z"

  fAPickRef(A());
  // CHECK: call {{.*}} @"?fAPickRef@@YAXAEAUA@@@Z"

  fAPickRef2(A());
  // CHECK: call {{.*}} @"?fAPickRef2@@YAXAEAUA@@@Z"

  NS::fAPickConstRef(A());
  // CHECK: call {{.*}} @"?fAPickConstRef@NS@@YAXAEBUA@@@Z"

  NS::fBPickConstRef(B());
  // CHECK: call {{.*}} @"?fBPickConstRef@NS@@YAXAEBUA@@@Z"

  NS::fAPickRef(A());
  // CHECK: call {{.*}} @"?fAPickRef@NS@@YAXAEAUA@@@Z"

  NS::fAPickRef2(A());
  // CHECK: call {{.*}} @"?fAPickRef2@NS@@YAXAEAUA@@@Z"

  S::fAPickConstRef(A());
  // CHECK: call {{.*}} @"?fAPickConstRef@S@@SAXAEBUA@@@Z"
}

void test_member_call() {
  S s;

  static_cast<S&&>(s).memberPickNonConst();
  // CHECK: call {{.*}} @"?memberPickNonConst@S@@QEAAXXZ"

  static_cast<S&&>(s).memberPickConstRef();
  // CHECK: call {{.*}} @"?memberPickConstRef@S@@QEGBAXXZ"
}
