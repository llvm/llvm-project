// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple %itanium_abi_triple | FileCheck %s --check-prefixes=CHECK,ITANIUM
// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple x86_64-pc-win32 2>&1 | FileCheck %s --check-prefixes=CHECK,MSABI

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
}

struct Primary {
  virtual void f();
  std::strong_ordering operator<=>(const Primary&) const = default;
};
struct X {
  virtual struct Y &operator=(Y&&);
  virtual struct Y &operator=(const Y&);
  std::strong_ordering operator<=>(const X&) const = default;
};
// The vtable for Y should contain the following entries in order:
//  - Primary::f
//  - Y::operator<=>
//  - Y::operator=(const Y&) (implicit)
//  - Y::operator=(Y&&) (implicit)
//  - Y::operator==(const Y&) const (implicit)
// See:
//   https://github.com/itanium-cxx-abi/cxx-abi/issues/83 for assignment operator
//   https://github.com/itanium-cxx-abi/cxx-abi/issues/88 for equality comparison
// FIXME: What rule does MSVC use?
struct Y : Primary, X {
  virtual std::strong_ordering operator<=>(const Y&) const = default;
};
Y y;
// ITANIUM: @_ZTV1Y = {{.*}}constant {{.*}} null, {{.*}} @_ZTI1Y, {{.*}} @_ZN7Primary1fEv, {{.*}} @_ZNK1YssERKS_, {{.*}} @_ZN1YaSERKS_, {{.*}} @_ZN1YaSEOS_, {{.*}} @_ZNK1YeqERKS_], {{.*}} -[[POINTERSIZE:4|8]]
// ITANIUM-SAME: @_ZTI1Y, {{.*}} @_ZThn[[POINTERSIZE]]_N1YaSERKS_

struct A {
  void operator<=>(int);
};

// ITANIUM: define {{.*}}@_ZN1AssEi(
// MSABI: define {{.*}}@"??__MA@@QEAAXH@Z"(
void A::operator<=>(int) {}

// ITANIUM: define {{.*}}@_Zssi1A(
// MSABI: define {{.*}}@"??__M@YAXHUA@@@Z"(
void operator<=>(int, A) {}

int operator<=>(A, A);

// ITANIUM: define {{.*}}_Z1f1A(
// MSABI: define {{.*}}@"?f@@YAHUA@@@Z"(
int f(A a) {
  // ITANIUM: %[[RET:.*]] = call {{.*}}_Zss1AS_(
  // ITANIUM: ret i32 %[[RET]]
  // MSABI: %[[RET:.*]] = call {{.*}}"??__M@YAHUA@@0@Z"(
  // MSABI: ret i32 %[[RET]]
  return a <=> a;
}

// CHECK-LABEL: define {{.*}}builtin_cmp
void builtin_cmp(int a) {
  // CHECK: icmp slt
  // CHECK: select
  // CHECK: icmp eq
  // CHECK: select
  a <=> a;
}

namespace GH140368 {
union U {
  bool operator==(const U&) const = default;
  std::strong_ordering operator<=>(const U&) const = default;
};
template<typename T> union TU {
  bool operator==(const TU&) const = default;
};

// ITANIUM-LABEL: define {{.*}}@_ZN8GH1403682eqERKNS_1UES2_(
// MSABI-LABEL: define {{.*}}@"?eq@GH140368@@YA_NAEBTU@1@0@Z"(
bool eq(const U &a, const U &b) {
  // ITANIUM: %[[RET:.*]] = call {{.*}}@_ZNK8GH1403681UeqERKS0_(
  // ITANIUM: ret i1 %[[RET]]
  // MSABI: %[[RET:.*]] = call {{.*}}@"??8U@GH140368@@QEBA_NAEBT01@@Z"(
  // MSABI: ret i1 %[[RET]]
  return a == b;
}

// ITANIUM-LABEL: define linkonce_odr {{.*}}@_ZNK8GH1403681UeqERKS0_(
// ITANIUM: ret i1 true
// MSABI-LABEL: define linkonce_odr {{.*}}@"??8U@GH140368@@QEBA_NAEBT01@@Z"(
// MSABI: ret i1 true

// ITANIUM-LABEL: define {{.*}}@_ZN8GH1403682neERKNS_1UES2_(
// MSABI-LABEL: define {{.*}}@"?ne@GH140368@@YA_NAEBTU@1@0@Z"(
bool ne(const U &a, const U &b) {
  // ITANIUM: %[[EQ:.*]] = call {{.*}}@_ZNK8GH1403681UeqERKS0_(
  // ITANIUM: xor i1 %[[EQ]], true
  // MSABI: %[[EQ:.*]] = call {{.*}}@"??8U@GH140368@@QEBA_NAEBT01@@Z"(
  // MSABI: xor i1 %[[EQ]], true
  return a != b;
}

// ITANIUM-LABEL: define {{.*}}@_ZN8GH1403683cmpERKNS_1UES2_(
// MSABI-LABEL: define {{.*}}@"?cmp@GH140368@@YAHAEBTU@1@0@Z"(
int cmp(const U &a, const U &b) {
  // ITANIUM: call {{.*}}@_ZNK8GH1403681UssERKS0_(
  // MSABI: call {{.*}}@"??__MU@GH140368@@QEBA?AUstrong_ordering@std@@AEBT01@@Z"(
  return a <=> b;
}

// ITANIUM-LABEL: define {{.*}}@_ZN8GH1403682ltERKNS_1UES2_(
// MSABI-LABEL: define {{.*}}@"?lt@GH140368@@YA_NAEBTU@1@0@Z"(
bool lt(const U &a, const U &b) {
  // ITANIUM: call {{.*}}@_ZNK8GH1403681UssERKS0_(
  // ITANIUM: icmp slt
  // MSABI: call {{.*}}@"??__MU@GH140368@@QEBA?AUstrong_ordering@std@@AEBT01@@Z"(
  // MSABI: icmp slt
  return a < b;
}

// ITANIUM-LABEL: define {{.*}}@_ZN8GH1403683teqERKNS_2TUIiEES3_(
// MSABI-LABEL: define {{.*}}@"?teq@GH140368@@YA_NAEBT?$TU@H@1@0@Z"(
bool teq(const TU<int> &a, const TU<int> &b) {
  // ITANIUM: %[[RET:.*]] = call {{.*}}@_ZNK8GH1403682TUIiEeqERKS1_(
  // ITANIUM: ret i1 %[[RET]]
  // MSABI: %[[RET:.*]] = call {{.*}}@"??8?$TU@H@GH140368@@QEBA_NAEBT01@@Z"(
  // MSABI: ret i1 %[[RET]]
  return a == b;
}
}
