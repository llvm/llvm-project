// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=NORMAL
// RUN: %clang_cc1 %s -std=c++11 -fms-compatibility -triple=x86_64-pc-win32 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=MSVCCOMPAT
// CHECK: ; ModuleID 

struct A {
    inline void f();
};

// NORMAL-NOT: define{{.*}} void @_ZN1A1fEv
// MSVCCOMPAT-NOT: define{{.*}} void @"?f@A@@QEAAXXZ"
void A::f() { }

template<typename> struct B { };

template<> struct B<char> {
  inline void f();
};

// NORMAL-NOT: _ZN1BIcE1fEv
// MSVCCOMPAT-NOT: @"?f@?$B@D@@QEAAXXZ"
void B<char>::f() { }

// We need a final CHECK line here.

// NORMAL-LABEL: define{{.*}} void @_Z1fv
// MSVCCOMPAT-LABEL: define dso_local void @"?f@@YAXXZ"
void f() { }

// <rdar://problem/8740363>
inline void f1(int);

// NORMAL-LABEL: define linkonce_odr void @_Z2f1i
// MSVCCOMPAT-LABEL: define linkonce_odr dso_local void @"?f1@@YAXH@Z"
void f1(int) { }

void test_f1() { f1(17); }

// PR8789
namespace test1 {
  template <typename T> class ClassTemplate {
  private:
    friend void T::func();
    void g() {}
  };

  // NORMAL-LABEL: define linkonce_odr void @_ZN5test11C4funcEv(
  // MSVCCOMPAT-LABEL: define linkonce_odr dso_local void @"?func@C@test1@@QEAAXXZ"(

  class C {
  public:
    void func() {
      ClassTemplate<C> ct;
      ct.g();
    }
  };

  void f() {
    C c;
    c.func();
  }
}

// PR13252
namespace test2 {
  struct A;
  void f(const A& a);
  struct A {
    friend void f(const A& a) { } 
  };
  void g() {
    A a;
    f(a);
  }
  // NORMAL-LABEL: define linkonce_odr void @_ZN5test21fERKNS_1AE
  // MSVCCOMPAT-LABEL: define linkonce_odr dso_local void @"?f@test2@@YAXAEBUA@1@@Z"
}

// NORMAL-NOT: _Z17ExternAndInlineFnv
// MSVCCOMPAT-LABEL: define weak_odr dso_local void @"?ExternAndInlineFn@@YAXXZ"
extern inline void ExternAndInlineFn() {}

// NORMAL-NOT: _Z18InlineThenExternFnv
// MSVCCOMPAT-LABEL: define weak_odr dso_local void @"?InlineThenExternFn@@YAXXZ"
inline void InlineThenExternFn() {}
extern void InlineThenExternFn();

// NORMAL-LABEL: define{{.*}} void @_Z18ExternThenInlineFnv
// MSVCCOMPAT-LABEL: define dso_local void @"?ExternThenInlineFn@@YAXXZ"
extern void ExternThenInlineFn() {}

// NORMAL-NOT: _Z25ExternThenInlineThenDefFnv
// MSVCCOMPAT-LABEL: define weak_odr dso_local void @"?ExternThenInlineThenDefFn@@YAXXZ"
extern void ExternThenInlineThenDefFn();
inline void ExternThenInlineThenDefFn();
void ExternThenInlineThenDefFn() {}

// NORMAL-NOT: _Z25InlineThenExternThenDefFnv
// MSVCCOMPAT-LABEL: define weak_odr dso_local void @"?InlineThenExternThenDefFn@@YAXXZ"
inline void InlineThenExternThenDefFn();
extern void InlineThenExternThenDefFn();
void InlineThenExternThenDefFn() {}

// NORMAL-NOT: _Z17ExternAndConstexprFnv
// MSVCCOMPAT-LABEL: define weak_odr dso_local i32 @"?ExternAndConstexprFn@@YAHXZ"
extern constexpr int ExternAndConstexprFn() { return 0; }

// NORMAL-NOT: _Z11ConstexprFnv
// MSVCCOMPAT-NOT: @"?ConstexprFn@@YAHXZ"
constexpr int ConstexprFn() { return 0; }

template <typename T>
extern inline void ExternInlineOnPrimaryTemplate(T);

// NORMAL-LABEL: define{{.*}} void @_Z29ExternInlineOnPrimaryTemplateIiEvT_
// MSVCCOMPAT-LABEL: define dso_local void @"??$ExternInlineOnPrimaryTemplate@H@@YAXH@Z"
template <>
void ExternInlineOnPrimaryTemplate(int) {}

template <typename T>
extern inline void ExternInlineOnPrimaryTemplateAndSpecialization(T);

// NORMAL-NOT: _Z46ExternInlineOnPrimaryTemplateAndSpecializationIiEvT_
// MSVCCOMPAT-LABEL: define weak_odr dso_local void @"??$ExternInlineOnPrimaryTemplateAndSpecialization@H@@YAXH@Z"
template <>
extern inline void ExternInlineOnPrimaryTemplateAndSpecialization(int) {}

struct TypeWithInlineMethods {
  // NORMAL-NOT: _ZN21TypeWithInlineMethods9StaticFunEv
  // MSVCCOMPAT-NOT: @"?StaticFun@TypeWithInlineMethods@@SAXXZ"
  static void StaticFun() {}
  // NORMAL-NOT: _ZN21TypeWithInlineMethods12NonStaticFunEv
  // MSVCCOMPAT-NOT: @"?NonStaticFun@TypeWithInlineMethods@@QEAAXXZ"
  void NonStaticFun() { StaticFun(); }
};

namespace PR22959 {
template <typename>
struct S;

S<int> Foo();

template <typename>
struct S {
  friend S<int> Foo();
};

__attribute__((used)) inline S<int> Foo() { return S<int>(); }
// NORMAL-LABEL: define linkonce_odr void @_ZN7PR229593FooEv(
// MSVCCOMPAT-LABEL: define linkonce_odr dso_local i8 @"?Foo@PR22959@@YA?AU?$S@H@1@XZ"(
}
