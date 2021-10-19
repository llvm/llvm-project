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
           bf0 &&                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 10
           bt1 &&                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 8
           bf1 &&                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 6
           bt2 &&                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 4
           bf2;                     // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 2

  bool b = bt0 ||
           bf0 ||                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 20
           bt1 ||                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 18
           bf1 ||                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 16
           bt2 ||                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 14
           bf2;                     // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 12

  bool c = (bt0  &&
            bf0) ||                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 27
           (bt1  &&
            bf1) ||                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 29
           (bt2  &&
            bf2) ||                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 31
           (bt3  &&
            bf3) ||                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 33
           (bt4  &&
            bf4) ||                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 35
           (bf5  &&
            bf5);                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 37

  bool d = (bt0  ||
            bf0) &&                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 44
           (bt1  ||
            bf1) &&                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 46
           (bt2  ||
            bf2) &&                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 48
           (bt3  ||
            bf3) &&                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 50
           (bt4  ||
            bf4) &&                 // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 52
           (bt5  ||
            bf5);                   // CHECK: store {{.*}} @[[FUNC]], i64 0, i64 54

  return a && b && c && d;
}
