// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

struct Base {
  virtual ~Base();
} ;

struct Derived : Base {
  void operator delete(void *) { }
  Derived();
};

void foo() {
  Derived d1;			// ok
}

// CHECK-LABEL: define internal i32 @_Z1fj(
inline unsigned f(unsigned n) { return n == 0 ? 0 : n + f(n-1); }

unsigned g(unsigned n) { return f(n); }

// rdar://problem/10133200: give explicit instantiations external linkage in kernel mode
// CHECK-LABEL: define{{.*}} void @_Z3barIiEvv()
template <typename T> void bar() {}
template void bar<int>();

// CHECK-LABEL: define internal i32 @_Z5identIiET_S0_(
template <typename X> X ident(X x) { return x; }

int foo(int n) { return ident(n); }

// CHECK-LABEL: define internal void @_ZN7DerivedD1Ev(
// CHECK-LABEL: define internal void @_ZN7DerivedD0Ev(
// CHECK-LABEL: define internal void @_ZN7DeriveddlEPv(
