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
           bf0 && // CHECK: store {{.*}} @[[FUNC]], i64 80
           bt1 && // CHECK: store {{.*}} @[[FUNC]], i64 64
           bf1 && // CHECK: store {{.*}} @[[FUNC]], i64 48
           bt2 && // CHECK: store {{.*}} @[[FUNC]], i64 32
           bf2;   // CHECK: store {{.*}} @[[FUNC]], i64 16

  bool b = bt0 ||
           bf0 || // CHECK: store {{.*}} @[[FUNC]], i64 160
           bt1 || // CHECK: store {{.*}} @[[FUNC]], i64 144
           bf1 || // CHECK: store {{.*}} @[[FUNC]], i64 128
           bt2 || // CHECK: store {{.*}} @[[FUNC]], i64 112
           bf2;   // CHECK: store {{.*}} @[[FUNC]], i64 96

  bool c = (bt0 &&
            bf0) || // CHECK: store {{.*}} @[[FUNC]], i64 216
           (bt1 &&
            bf1) || // CHECK: store {{.*}} @[[FUNC]], i64 232
           (bt2 &&
            bf2) || // CHECK: store {{.*}} @[[FUNC]], i64 248
           (bt3 &&
            bf3) || // CHECK: store {{.*}} @[[FUNC]], i64 264
           (bt4 &&
            bf4) || // CHECK: store {{.*}} @[[FUNC]], i64 280
           (bf5 &&
            bf5); // CHECK: store {{.*}} @[[FUNC]], i64 296

  bool d = (bt0 ||
            bf0) && // CHECK: store {{.*}} @[[FUNC]], i64 352
           (bt1 ||
            bf1) && // CHECK: store {{.*}} @[[FUNC]], i64 368
           (bt2 ||
            bf2) && // CHECK: store {{.*}} @[[FUNC]], i64 384
           (bt3 ||
            bf3) && // CHECK: store {{.*}} @[[FUNC]], i64 400
           (bt4 ||
            bf4) && // CHECK: store {{.*}} @[[FUNC]], i64 416
           (bt5 ||
            bf5); // CHECK: store {{.*}} @[[FUNC]], i64 432

  return a && b && c && d;
}
