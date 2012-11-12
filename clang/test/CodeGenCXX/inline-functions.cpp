// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
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

// CHECK: define void @_Z1fv
void f() { }

// <rdar://problem/8740363>
inline void f1(int);

// CHECK: define linkonce_odr void @_Z2f1i
void f1(int) { }

void test_f1() { f1(17); }

// PR8789
namespace test1 {
  template <typename T> class ClassTemplate {
  private:
    friend void T::func();
    void g() {}
  };

  // CHECK: define linkonce_odr void @_ZN5test11C4funcEv(

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
  // CHECK: define linkonce_odr void @_ZN5test21fERKNS_1AE
}
