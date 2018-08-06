// RUN: %clang_cc1 -triple x86_64-pc-linux -mrelocation-model static -O1 -emit-llvm %s -o - | FileCheck --check-prefix=STATIC %s
// RUN: %clang_cc1 -triple x86_64-pc-linux -mrelocation-model static -fno-plt -O1 -emit-llvm %s -o - | FileCheck --check-prefix=NOPLT %s

// STATIC-DAG: @_ZTV1C = linkonce_odr dso_local unnamed_addr constant
// STATIC-DAG: @_ZTS1C = linkonce_odr dso_local constant
// STATIC-DAG: @_ZTI1C = linkonce_odr dso_local constant
// STATIC-DAG: @_ZZ14useStaticLocalvE3obj = linkonce_odr dso_local global
// STATIC-DAG: @_ZGVZN5guard1gEvE1a = linkonce_odr dso_local global
// STATIC-DAG: define dso_local void @_ZN1CC2Ev(
// STATIC-DAG: define dso_local void @_ZN1CC1Ev(
// STATIC-DAG: define linkonce_odr dso_local void @_ZN1C3fooEv(

// NOPLT-DAG: @_ZTV1C = linkonce_odr dso_local unnamed_addr constant
// NOPLT-DAG: @_ZTS1C = linkonce_odr dso_local constant
// NOPLT-DAG: @_ZTI1C = linkonce_odr dso_local constant
// NOPLT-DAG: @_ZZ14useStaticLocalvE3obj = linkonce_odr dso_local global
// NOPLT-DAG: @_ZGVZN5guard1gEvE1a = linkonce_odr dso_local global
// NOPLT-DAG: define dso_local void @_ZN1CC2Ev(
// NOPLT-DAG: define dso_local void @_ZN1CC1Ev(
// NOPLT-DAG: define linkonce_odr dso_local void @_ZN1C3fooEv(

struct C {
  C();
  virtual void foo() {}
};
C::C() {}

struct HasVTable {
  virtual void f();
};
inline HasVTable &useStaticLocal() {
  static HasVTable obj;
  return obj;
}
void useit() {
  useStaticLocal();
}

namespace guard {
int f();
inline int g() {
  static int a = f();
  return a;
}
int h() {
  return g();
}
} // namespace guard


// STATIC-DAG: define available_externally dso_local void @_ZN5test23barIcEC1Ev(
// NOPLT-DAG: define available_externally void @_ZN5test23barIcEC1Ev(
namespace test2 {
void foo();
template <typename T>
struct bar {
  virtual void zed();
  bar() { foo(); }
};
extern template class bar<char>;
bar<char> abc;
} // namespace test2
