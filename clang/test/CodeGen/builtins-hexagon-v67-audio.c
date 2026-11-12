// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv67 -target-feature +audio -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv67t -target-feature +audio -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @test1
// CHECK: call i64 @llvm.hexagon.M7.dcmpyrw(i64 %0, i64 %1)
long long test1(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyrw(rss, rtt);
}

// CHECK-LABEL: @test2
// CHECK: call i64 @llvm.hexagon.M7.dcmpyrw.acc(i64 %0, i64 %1, i64 %2)
long long test2(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyrw_acc(rxx, rss, rtt);
}

// CHECK-LABEL: @test3
// CHECK: call i64 @llvm.hexagon.M7.dcmpyrwc(i64 %0, i64 %1)
long long test3(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyrwc(rss, rtt);
}

// CHECK-LABEL: @test4
// CHECK: call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %0, i64 %1, i64 %2)
long long test4(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyrwc_acc(rxx, rss, rtt);
}

// CHECK-LABEL: @test5
// CHECK: call i64 @llvm.hexagon.M7.dcmpyiw(i64 %0, i64 %1)
long long test5(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyiw(rss, rtt);
}

// CHECK-LABEL: @test6
// CHECK: call i64 @llvm.hexagon.M7.dcmpyiw.acc(i64 %0, i64 %1, i64 %2)
long long test6(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyiw_acc(rxx, rss, rtt);
}

// CHECK-LABEL: @test7
// CHECK: call i64 @llvm.hexagon.M7.dcmpyiwc(i64 %0, i64 %1)
long long test7(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyiwc(rss, rtt);
}

// CHECK-LABEL: @test8
// CHECK: call i64 @llvm.hexagon.M7.dcmpyiwc.acc(i64 %0, i64 %1, i64 %2)
long long test8(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_dcmpyiwc_acc(rxx, rss, rtt);
}

// CHECK-LABEL: @test9
// CHECK: call i32 @llvm.hexagon.M7.wcmpyrw(i64 %0, i64 %1)
int test9(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyrw(rss, rtt);
}

// CHECK-LABEL: @test10
// CHECK: call i32 @llvm.hexagon.M7.wcmpyrwc(i64 %0, i64 %1)
int test10(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyrwc(rss, rtt);
}

// CHECK-LABEL: @test11
// CHECK: call i32 @llvm.hexagon.M7.wcmpyiw(i64 %0, i64 %1)
int test11(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyiw(rss, rtt);
}

// CHECK-LABEL: @test12
// CHECK: call i32 @llvm.hexagon.M7.wcmpyiwc(i64 %0, i64 %1)
int test12(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyiwc(rss, rtt);
}

// CHECK-LABEL: @test13
// CHECK: call i32 @llvm.hexagon.M7.wcmpyrw.rnd(i64 %0, i64 %1)
int test13(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyrw_rnd(rss, rtt);
}

// CHECK-LABEL: @test14
// CHECK: call i32 @llvm.hexagon.M7.wcmpyrwc.rnd(i64 %0, i64 %1)
int test14(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyrwc_rnd(rss, rtt);
}

// CHECK-LABEL: @test15
// CHECK: call i32 @llvm.hexagon.M7.wcmpyiw.rnd(i64 %0, i64 %1)
int test15(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyiw_rnd(rss, rtt);
}

// CHECK-LABEL: @test16
// CHECK: call i32 @llvm.hexagon.M7.wcmpyiwc.rnd(i64 %0, i64 %1)
int test16(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_wcmpyiwc_rnd(rss, rtt);
}

// CHECK-LABEL: @test17
// CHECK: call i64 @llvm.hexagon.A7.croundd.ri(i64 %0, i32 0)
long long test17(long long rss) {
  return __builtin_HEXAGON_A7_croundd_ri(rss, 0);
}

// CHECK-LABEL: @test18
// CHECK: call i64 @llvm.hexagon.A7.croundd.rr(i64 %0, i32 %1)
long long test18(long long rss, int rt) {
  return __builtin_HEXAGON_A7_croundd_rr(rss, rt);
}

// CHECK-LABEL: @test19
// CHECK: call i32 @llvm.hexagon.A7.clip(i32 %0, i32 0)
int test19(int rs) {
  return __builtin_HEXAGON_A7_clip(rs, 0);
}

// CHECK-LABEL: @test20
// CHECK: call i64 @llvm.hexagon.A7.vclip(i64 %0, i32 0)
long long test20(long long rs) {
  return __builtin_HEXAGON_A7_vclip(rs, 0);
}

// CHECK-LABEL: @test21
// CHECK: call i64 @llvm.hexagon.M7.vdmpy(i64 %0, i64 %1)
long long test21(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_vdmpy(rss, rtt);
}

// CHECK-LABEL: @test22
// CHECK: call i64 @llvm.hexagon.M7.vdmpy.acc(i64 %0, i64 %1, i64 %2)
long long test22(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_vdmpy_acc(rxx, rss, rtt);
}

