// RUN: %clang_cc1 -triple x86_64-windows-msvc %s -emit-llvm -fms-extensions -fms-compatibility -fms-reference-binding -Wno-microsoft-reference-binding -o - | FileCheck %s

struct A {};
struct B : A {};

void fAPickConstRef(A&) {}
void fAPickConstRef(const A&) {}

void fBPickConstRef(A&) {}
void fBPickConstRef(const A&) {}

void fAPickRef(A&) {}
void fAPickRef(const volatile A&) {}

// NOTE: MSVC incorrectly picks the `const volatile A&` overload with the mangled name
// "?fBPickConstVolatileRef@@YAXAEDUA@@@Z" when converting a derived class `B` to base `A`.
// This occurs even with conforming reference binding enabled so this isn't a one off
// behaviour with non-conforming MSVC specific reference binding but appears to be a bug.
// A bug report has been submitted to MSVC. This bug has been verified upto MSVC 19.40.
// We are not emulating this behaviour and instead will pick the `A&` overload as intended.
void fBPickConstVolatileRef(A&) {}
void fBPickConstVolatileRef(const volatile A&) {}

namespace NS {
  void fAPickConstRef(A&) {}
  void fAPickConstRef(const A&) {}

  void fBPickConstRef(A&) {}
  void fBPickConstRef(const A&) {}

  void fAPickRef(A&) {}
  void fAPickRef(const volatile A&) {}

  // See the above note above the global `fBPickConstVolatileRef`
  void fBPickConstVolatileRef(A&) {}
  void fBPickConstVolatileRef(const volatile A&) {}
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

  fBPickConstVolatileRef(B());
  // CHECK: call {{.*}} @"?fBPickConstVolatileRef@@YAXAEAUA@@@Z"

  NS::fAPickConstRef(A());
  // CHECK: call {{.*}} @"?fAPickConstRef@NS@@YAXAEBUA@@@Z"

  NS::fBPickConstRef(B());
  // CHECK: call {{.*}} @"?fBPickConstRef@NS@@YAXAEBUA@@@Z"

  NS::fAPickRef(A());
  // CHECK: call {{.*}} @"?fAPickRef@NS@@YAXAEAUA@@@Z"

  NS::fBPickConstVolatileRef(B());
  // CHECK: call {{.*}} @"?fBPickConstVolatileRef@NS@@YAXAEAUA@@@Z"

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
