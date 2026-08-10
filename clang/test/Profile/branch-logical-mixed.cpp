// Test to ensure instrumentation of logical operator RHS True/False counters
// are being instrumented for branch coverage

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -main-file-name branch-logical-mixed.cpp %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck -allow-deprecated-dag-overlap %s


// CHECK: @[[FUNC:__profc__Z4funcv]] = {{.*}} global [61 x i64] zeroinitializer


// CHECK-LABEL: @_Z4funcv()
bool func() {
  bool bt0 = true;
  bool bt1 = true;
  bool bt2 = true;
  bool bt3 = true;
  bool bt4 = true;
  bool bt5 = true;
  bool bf0 = false;
  bool bf1 = false;
  bool bf2 = false;
  bool bf3 = false;
  bool bf4 = false;
  bool bf5 = false;

  bool a = bt0 &&
           bf0 && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 10
           bt1 && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 8
           bf1 && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 6
           bt2 && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 4
           bf2;   // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 2

  bool b = bt0 ||
           bf0 || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 20
           bt1 || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 18
           bf1 || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 16
           bt2 || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 14
           bf2;   // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 12

  bool c = (bt0 &&
            bf0) || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 27
           (bt1 &&
            bf1) || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 29
           (bt2 &&
            bf2) || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 31
           (bt3 &&
            bf3) || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 33
           (bt4 &&
            bf4) || // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 35
           (bf5 &&
            bf5); // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 37

  bool d = (bt0 ||
            bf0) && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 44
           (bt1 ||
            bf1) && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 46
           (bt2 ||
            bf2) && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 48
           (bt3 ||
            bf3) && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 50
           (bt4 ||
            bf4) && // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 52
           (bt5 ||
            bf5); // CHECK: store {{.*}} @[[FUNC]], i32 0, i32 54

  return a && b && c && d;
}
