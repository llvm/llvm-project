// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -UDESUGAR %s | opt -passes=instnamer -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -DDESUGAR %s | opt -passes=instnamer -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -UDESUGAR -DTEMPLATE %s | opt -passes=instnamer -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -DDESUGAR -DTEMPLATE %s | opt -passes=instnamer -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -UDESUGAR -DTEMPLATE -DDEPENDENT %s | opt -passes=instnamer -S | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c++11 -emit-llvm -o - -DDESUGAR -DTEMPLATE -DDEPENDENT %s | opt -passes=instnamer -S | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
};

struct B {
  B();
  B(const B &);
  ~B();
};

struct C {
  C(const B &);
  C(const C &);
  ~C();
};

struct E;
struct D {
  D(const C &);
  D(const D &);
  ~D();
};
E begin(D);
E end(D);

struct F;
struct G;
struct H;
struct E {
  E(const E &);
  ~E();
  F operator*();
  G operator++();
  H operator!=(const E &o);
};

struct I;
struct F {
  F(const F &);
  ~F();
  operator I();
};

struct G {
  G(const G &);
  ~G();
  operator bool();
};

struct H {
  H(const H &);
  ~H();
  operator bool();
};

struct I {
  I(const I &);
  ~I();
};

void body(const I &);

#ifdef TEMPLATE
#ifdef DEPENDENT
template<typename D>
#else
template<typename>
#endif
#endif
void for_temps() {
  A a;
#ifdef DESUGAR
  {
    auto && __range = D(B());
    for (auto __begin = begin(__range), __end = end(__range);
         __begin != __end; ++__begin) {
      I i = *__begin;
      body(i);
    }
  }
#else
  for (I i : D(B())) {
    body(i);
  }
#endif
}

#ifdef TEMPLATE
template void for_temps<D>();
#endif

// CHECK: define {{.*}}for_temps
// CHECK: call void @_ZN1AC1Ev(
// CHECK: call void @_ZN1BC1Ev(
// CHECK: call void @_ZN1CC1ERK1B(
// CHECK: call void @_ZN1DC1ERK1C(
// CHECK: call void @_ZN1CD1Ev(
// CHECK: call void @_ZN1BD1Ev(
// CHECK: call void @_ZN1DC1ERKS_(
// CHECK: call void @_Z5begin1D(
// CHECK: call void @_ZN1DD1Ev(
// CHECK: call void @_ZN1DC1ERKS_(
// CHECK: call void @_Z3end1D(
// CHECK: call void @_ZN1DD1Ev(
// CHECK: br label %[[COND:.*]]

// CHECK: [[COND]]:
// CHECK: call void @_ZN1EneERKS_(
// CHECK: %[[CMP:.*]] = call noundef zeroext i1 @_ZN1HcvbEv(
// CHECK: call void @_ZN1HD1Ev(
// CHECK: br i1 %[[CMP]], label %[[BODY:.*]], label %[[CLEANUP:.*]]

// CHECK: [[CLEANUP]]:
// CHECK: call void @_ZN1ED1Ev(
// CHECK: call void @_ZN1ED1Ev(
// In for-range:
// call void @_ZN1DD1Ev(
// CHECK: br label %[[END:.*]]

// CHECK: [[BODY]]:
// CHECK: call void @_ZN1EdeEv(
// CHECK: call void @_ZN1Fcv1IEv(
// CHECK: call void @_ZN1FD1Ev(
// CHECK: call void @_Z4bodyRK1I(
// CHECK: call void @_ZN1ID1Ev(
// CHECK: br label %[[INC:.*]]

// CHECK: [[INC]]:
// CHECK: call void @_ZN1EppEv(
// CHECK: call void @_ZN1GD1Ev(
// CHECK: br label %[[COND]]

// CHECK: [[END]]:
// In desugared version:
// call void @_ZN1DD1Ev(
// CHECK: call void @_ZN1AD1Ev(
// CHECK: ret void
