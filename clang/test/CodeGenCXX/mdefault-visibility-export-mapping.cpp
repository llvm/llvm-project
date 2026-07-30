// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-DEF %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=none -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-DEF %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=explicit -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-DEF,EXPLICIT-EXP %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix %s -mdefault-visibility-export-mapping=all -emit-llvm -o - | \
// RUN:   FileCheck -check-prefixes=CHECK,UNSPECIFIED-EXP,EXPLICIT-EXP %s

struct A {};

template <class T>
class B {
public:
  T x;
  B(T _x) : x(_x) {}
  ~B() {}
  void func(T x) {}
};

template <class T>
class __attribute__((visibility("default"))) C {
public:
  T x;
  C(T _x) : x(_x) {}
  ~C() {}
  void func(T x) {}
};

class D {
public:
  ~D();
};

D::~D() {}

extern template class B<int>;
extern template class C<int>;

void func() {
  B<A> x({});
  C<A> y({});
  x.func({});
  y.func({});
  B<int> xi(0);
  C<int> yi(0);
  xi.func(0);
  yi.func(0);
  D z;
}

// D::~D() (base object destructor)
// UNSPECIFIED-DEF:  define void @_ZN1DD2Ev(
// UNSPECIFIED-EXP:  define dllexport void @_ZN1DD2Ev(

// D::~D() (complete object destructor)
// UNSPECIFIED-DEF:  define void @_ZN1DD1Ev(
// UNSPECIFIED-EXP:  define dllexport void @_ZN1DD1Ev(

// UNSPECIFIED-DEF: define void @_Z4funcv(
// UNSPECIFIED-EXP: define dllexport void @_Z4funcv(

// B<A>::B(A) (complete object constructor)
// UNSPECIFIED-DEF: define linkonce_odr void @_ZN1BI1AEC1ES0_(
// UNSPECIFIED-EXP: define linkonce_odr dllexport void @_ZN1BI1AEC1ES0_(

// C<A>::C(A) (complete object constructor)
// EXPLICIT-DEF: define linkonce_odr void @_ZN1CI1AEC1ES0_(
// EXPLICIT-EXP: define linkonce_odr dllexport void @_ZN1CI1AEC1ES0_(

// B<A>::func(A)
// UNSPECIFIED-DEF: define linkonce_odr void @_ZN1BI1AE4funcES0_(
// UNSPECIFIED-EXP: define linkonce_odr dllexport void @_ZN1BI1AE4funcES0_(

// C<A>::func(A)
// EXPLICIT-DEF: define linkonce_odr void @_ZN1CI1AE4funcES0_(
// EXPLICIT-EXP: define linkonce_odr dllexport void @_ZN1CI1AE4funcES0_(

// B<int>::B(int) (complete object constructor)
// CHECK: declare void @_ZN1BIiEC1Ei

// C<int>::C(int) (complete object constructor)
// CHECK: declare void @_ZN1CIiEC1Ei

// B<int>::func(int)
// CHECK: declare void @_ZN1BIiE4funcEi

// C<int>::func(int)
// CHECK: declare void @_ZN1CIiE4funcEi

// C<int>::~C() (complete object destructor)
// CHECK: declare void @_ZN1CIiED1Ev

// B<int>::~B() (complete object destructor)
// CHECK: declare void @_ZN1BIiED1Ev

// C<A>::~c() (complete object destructor)
// EXPLICIT-DEF: define linkonce_odr void @_ZN1CI1AED1Ev(
// EXPLICIT-EXP: define linkonce_odr dllexport void @_ZN1CI1AED1Ev(

// B<A>::~B() (complete object destructor)
// UNSPECIFIED-DEF: define linkonce_odr void @_ZN1BI1AED1Ev(
// UNSPECIFIED-EXP: define linkonce_odr dllexport void @_ZN1BI1AED1Ev(

// B<A>::B(A) (base object constructor)
// UNSPECIFIED-DEF: define linkonce_odr void @_ZN1BI1AEC2ES0_(
// UNSPECIFIED-EXP: define linkonce_odr dllexport void @_ZN1BI1AEC2ES0_(

// B<A>::~B() (base object destructor)
// UNSPECIFIED-DEF: define linkonce_odr void @_ZN1BI1AED2Ev(
// UNSPECIFIED-EXP: define linkonce_odr dllexport void @_ZN1BI1AED2Ev(

// C<A>::C(A) (base object constructor)
// EXPLICIT-DEF: define linkonce_odr void @_ZN1CI1AEC2ES0_(
// EXPLICIT-EXP: define linkonce_odr dllexport void @_ZN1CI1AEC2ES0_(

// C<A>::~C() (base object destructor)
// EXPLICIT-DEF: define linkonce_odr void @_ZN1CI1AED2Ev(
// EXPLICIT-EXP: define linkonce_odr dllexport void @_ZN1CI1AED2Ev(
