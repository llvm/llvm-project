// RUN: %clang_cc1 -S -triple %itanium_abi_triple -fms-extensions -emit-llvm %s -o - | FileCheck %s

struct F {};

F operator*=(F &lhs, int rhs);

F operator++(F &lhs);

struct S {
  short _m;
  S(short _m) : _m(_m) {}

  void putM(short rhs) { _m = rhs; }
  short getM() { return _m; }

  __declspec(property(get = getM, put = putM)) short theData;
};

int test1a(int i) {
  S tmp(i);
  tmp.theData *= 2;
  return tmp.theData;
}

// CHECK-LABEL: define {{.*}} @_Z6test1ai(
// CHECK: call {{.*}} @_ZN1SC1Es(
// CHECK: call {{.*}} @_ZN1S4getMEv(
// CHECK: call {{.*}} @_ZN1S4putMEs(
// CHECK: call {{.*}} @_ZN1S4getMEv(

template <typename T>
int test1b(int i) {
  T tmp(i);
  tmp.theData *= 2;
  return tmp.theData;
}

template int test1b<S>(int);

// CHECK-LABEL: define {{.*}} @_Z6test1bI1SEii(
// CHECK: call {{.*}} @_ZN1SC1Es(
// CHECK: call {{.*}} @_ZN1S4getMEv(
// CHECK: call {{.*}} @_ZN1S4putMEs(
// CHECK: call {{.*}} @_ZN1S4getMEv(

int test2a(int i) {
  S tmp(i);
  ++tmp.theData;
  return tmp.theData;
}

// CHECK-LABEL: define {{.*}} i32 @_Z6test2ai(
// CHECK: call {{.*}} @_ZN1SC1Es(
// CHECK: call {{.*}} @_ZN1S4getMEv(
// CHECK: call {{.*}} @_ZN1S4putMEs(
// CHECK: call {{.*}} @_ZN1S4getMEv(

template <typename T>
int test2b(int i) {
  T tmp(i);
  ++tmp.theData;
  return tmp.theData;
}

template int test2b<S>(int);

// CHECK-LABEL: define {{.*}} i32 @_Z6test2bI1SEii(
// CHECK: call void @_ZN1SC1Es(
// CHECK: call {{.*}} @_ZN1S4getMEv(
// CHECK: call {{.*}} @_ZN1S4putMEs(
// CHECK: call {{.*}} @_ZN1S4getMEv(
