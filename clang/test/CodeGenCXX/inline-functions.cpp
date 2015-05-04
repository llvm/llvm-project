// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=NORMAL
// RUN: %clang_cc1 %s -std=c++11 -fms-compatibility -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=MSVCCOMPAT
// CHECK: ; ModuleID 

struct A {
    inline void f();
};

// CHECK-NOT: define void @_ZN1A1fEv
void A::f() { }

template<typename> struct B { };

template<> struct B<char> {
  inline void f();
};

// CHECK-NOT: _ZN1BIcE1fEv
void B<char>::f() { }

// We need a final CHECK line here.

// CHECK-LABEL: define void @_Z1fv
void f() { }

// <rdar://problem/8740363>
inline void f1(int);

// CHECK-LABEL: define linkonce_odr void @_Z2f1i
void f1(int) { }

void test_f1() { f1(17); }

// PR8789
namespace test1 {
  template <typename T> class ClassTemplate {
  private:
    friend void T::func();
    void g() {}
  };

  // CHECK-LABEL: define linkonce_odr void @_ZN5test11C4funcEv(

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
  // CHECK-LABEL: define linkonce_odr void @_ZN5test21fERKNS_1AE
}

// MSVCCOMPAT-LABEL: define weak_odr void @_Z17ExternAndInlineFnv
// NORMAL-NOT: _Z17ExternAndInlineFnv
extern inline void ExternAndInlineFn() {}

// MSVCCOMPAT-LABEL: define weak_odr void @_Z18InlineThenExternFnv
// NORMAL-NOT: _Z18InlineThenExternFnv
inline void InlineThenExternFn() {}
extern void InlineThenExternFn();

// CHECK-LABEL: define void @_Z18ExternThenInlineFnv
extern void ExternThenInlineFn() {}

// MSVCCOMPAT-LABEL: define weak_odr void @_Z25ExternThenInlineThenDefFnv
// NORMAL-NOT: _Z25ExternThenInlineThenDefFnv
extern void ExternThenInlineThenDefFn();
inline void ExternThenInlineThenDefFn();
void ExternThenInlineThenDefFn() {}

// MSVCCOMPAT-LABEL: define weak_odr void @_Z25InlineThenExternThenDefFnv
// NORMAL-NOT: _Z25InlineThenExternThenDefFnv
inline void InlineThenExternThenDefFn();
extern void InlineThenExternThenDefFn();
void InlineThenExternThenDefFn() {}

// MSVCCOMPAT-LABEL: define weak_odr i32 @_Z20ExternAndConstexprFnv
// NORMAL-NOT: _Z17ExternAndConstexprFnv
extern constexpr int ExternAndConstexprFn() { return 0; }

// CHECK-NOT: _Z11ConstexprFnv
constexpr int ConstexprFn() { return 0; }

template <typename T>
extern inline void ExternInlineOnPrimaryTemplate(T);

// CHECK-LABEL: define void @_Z29ExternInlineOnPrimaryTemplateIiEvT_
template <>
void ExternInlineOnPrimaryTemplate(int) {}

template <typename T>
extern inline void ExternInlineOnPrimaryTemplateAndSpecialization(T);

// MSVCCOMPAT-LABEL: define weak_odr void @_Z46ExternInlineOnPrimaryTemplateAndSpecializationIiEvT_
// NORMAL-NOT: _Z46ExternInlineOnPrimaryTemplateAndSpecializationIiEvT_
template <>
extern inline void ExternInlineOnPrimaryTemplateAndSpecialization(int) {}

struct TypeWithInlineMethods {
  // CHECK-NOT: _ZN21TypeWithInlineMethods9StaticFunEv
  static void StaticFun() {}
  // CHECK-NOT: _ZN21TypeWithInlineMethods12NonStaticFunEv
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
// CHECK-LABEL: define linkonce_odr void @_ZN7PR229593FooEv(
}
