// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | FileCheck %s

extern "C" int printf(...);

int count;

struct S {
  S() : iS(++count) { printf("S::S(%d)\n", iS); }
  ~S() { printf("S::~S(%d)\n", iS); }
  int iS;
};


S arr[2][1];
S s1;
S arr1[3];
static S sarr[4];

int main () {}
S arr2[2];
static S sarr1[4];
S s2;
S arr3[3];

// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit
// CHECK: call {{.*}} @__cxa_atexit

struct T {
  double d;
  int n;
  ~T();
};
T t[2][3] = { 1.0, 2, 3.0, 4, 5.0, 6, 7.0, 8, 9.0, 10, 11.0, 12 };

// CHECK: call {{.*}} @__cxa_atexit
// CHECK: getelementptr inbounds ({{.*}} bitcast {{.*}}* @t to %struct.T*), i64 6
// CHECK: call void @_ZN1TD1Ev
// CHECK: icmp eq {{.*}} @t
// CHECK: br i1 {{.*}}
