// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s

struct A {
  ~A();
};

struct B {
  ~B() throw(int);
};

struct C {
  B b;
  ~C() {}
};

struct D {
  ~D() noexcept(false);
};

struct E {
  D d;
  ~E() {}
};

void foo() {
  A a;
  C c;
  E e;
  // CHECK: invoke {{.*}} @_ZN1ED1Ev
  // CHECK: invoke {{.*}} @_ZN1CD1Ev
  // CHECK: call {{.*}} @_ZN1AD1Ev
}

struct F {
  D d;
  ~F();
};
F::~F() noexcept(false) {}

struct G {
  D d;
  ~G();
};
G::~G() {}

struct H {
  B b;
  ~H() throw(int);
};
H::~H() throw(int) {}

struct I {
  B b;
  ~I();
};
I::~I() {}

// Template variants.

template <typename T>
struct TA {
  ~TA();
};

template <typename T>
struct TB {
  ~TB() throw(int);
};

template <typename T>
struct TC {
  TB<T> b;
  ~TC() {}
};

template <typename T>
struct TD {
  ~TD() noexcept(false);
};

template <typename T>
struct TE {
  TD<T> d;
  ~TE() {}
};

void tfoo() {
  TA<int> a;
  TC<int> c;
  TE<int> e;
  // CHECK: invoke {{.*}} @_ZN2TEIiED1Ev
  // CHECK: invoke {{.*}} @_ZN2TCIiED1Ev
  // CHECK: call {{.*}} @_ZN2TAIiED1Ev
}

template <typename T>
struct TF {
  TD<T> d;
  ~TF();
};
template <typename T>
TF<T>::~TF() noexcept(false) {}

template <typename T>
struct TG {
  TD<T> d;
  ~TG();
};
template <typename T>
TG<T>::~TG() {}

template <typename T>
struct TH {
  TB<T> b;
  ~TH();
};
template <typename T>
TH<T>::~TH() {}

void tinst() {
  TF<int> f;
  TG<int> g;
  TH<int> h;
}
// CHECK: define linkonce_odr {{.*}} @_ZN2THIiED1Ev
// CHECK: _ZTIi
// CHECK: __cxa_call_unexpected

struct VX
{ virtual ~VX() {} };

struct VY : VX
{ virtual ~VY() {} };

template<typename T>
struct TVY : VX
{ virtual ~TVY() {} };


struct VA {
  B b;
  virtual ~VA() {}
};

struct VB : VA
{ virtual ~VB() {} };

template<typename T>
struct TVB : VA
{ virtual ~TVB() {} };

void tinst2() {
  TVY<int> tvy;
  TVB<int> tvb;
}

template <typename T>
struct Sw {
  T t;
  ~Sw() {}
};

void tsw() {
  Sw<int> swi;
  Sw<B> swb;
}
// CHECK-NOT: define linkonce_odr {{.*}} @_ZN2SwI1BED1Ev({{.*}} #
// CHECK: define linkonce_odr {{.*}} @_ZN2SwI1BED1Ev({{.*}}
// CHECK: _ZTIi
// CHECK: __cxa_call_unexpected
// CHECK: define linkonce_odr {{.*}} @_ZN2SwIiED1Ev({{.*}} [[ATTRGRP:#[0-9]+]]

template <typename T>
struct TVC : VX
{ virtual ~TVC(); };
template <typename T>
TVC<T>::~TVC() {}

// CHECK: attributes [[ATTRGRP]] = { noinline nounwind{{.*}} }
