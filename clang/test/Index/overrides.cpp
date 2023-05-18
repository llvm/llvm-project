struct A {
  virtual void f(int);
};

struct B {
  virtual void f(int);
  virtual void g();
};

struct C : B, A { 
  virtual void g();
};

struct D : C {
  virtual void f(int);
};

void C::g() {}

struct E {
  virtual void h() = 0;
  template <typename T> void i(T);
};

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: overrides.cpp:11:16: CXXMethod=g:11:16 (virtual) [Overrides @7:16] Extent=[11:3 - 11:19]
// CHECK: overrides.cpp:15:16: CXXMethod=f:15:16 (virtual) [Overrides @2:16, @6:16] Extent=[15:3 - 15:22]
// CHECK: overrides.cpp:18:9: CXXMethod=g:18:9 (Definition) (virtual) [Overrides @7:16] Extent=[18:1 - 18:15]
// CHECK: overrides.cpp:21:16: CXXMethod=h:21:16 (virtual) (pure) Extent=[21:3 - 21:23]
// CHECK: overrides.cpp:22:30: FunctionTemplate=i:22:30 Extent=[22:3 - 22:34]
