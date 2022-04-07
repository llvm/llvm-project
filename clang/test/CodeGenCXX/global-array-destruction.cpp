// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | FileCheck %s

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
// CHECK: getelementptr inbounds ([2 x [3 x %struct.T]], [2 x [3 x %struct.T]]* bitcast ([2 x [3 x { double, i32 }]]* @t to [2 x [3 x %struct.T]]*), i64 1, i64 0, i64 0)
// CHECK: call void @_ZN1TD1Ev
// CHECK: icmp eq {{.*}} @t
// CHECK: br i1 {{.*}}

static T t2[2][3] = { 1.0, 2, 3.0, 4, 5.0, 6, 7.0, 8, 9.0, 10, 11.0, 12 };

// CHECK: call {{.*}} @__cxa_atexit
// CHECK: getelementptr inbounds ([2 x [3 x %struct.T]], [2 x [3 x %struct.T]]* bitcast ([2 x [3 x { double, i32 }]]* @_ZL2t2 to [2 x [3 x %struct.T]]*), i64 1, i64 0, i64 0)
// CHECK: call void @_ZN1TD1Ev
// CHECK: icmp eq {{.*}} @_ZL2t2
// CHECK: br i1 {{.*}}

using U = T[2][3];
U &&u = U{ {{1.0, 2}, {3.0, 4}, {5.0, 6}}, {{7.0, 8}, {9.0, 10}, {11.0, 12}} };

// CHECK: call {{.*}} @__cxa_atexit
// CHECK: getelementptr inbounds ([2 x [3 x %struct.T]], [2 x [3 x %struct.T]]* @_ZGR1u_, i64 1, i64 0, i64 0)
// CHECK: call void @_ZN1TD1Ev
// CHECK: icmp eq {{.*}} @_ZGR1u_
// CHECK: br i1 {{.*}}
