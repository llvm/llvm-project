// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv67 -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @t1
// CHECK: call double @llvm.hexagon.F2.dfmpylh(double %0, double %1, double %2)
double t1(double a, double b, double c) {
  return __builtin_HEXAGON_F2_dfmpylh(a, b, c);
}

// CHECK-LABEL: @t2
// CHECK: call double @llvm.hexagon.F2.dfmpyhh(double %0, double %1, double %2)
double t2(double a, double b, double c) {
  return __builtin_HEXAGON_F2_dfmpyhh(a, b, c);
}

// CHECK-LABEL: @t3
// CHECK: call double @llvm.hexagon.F2.dfmax(double %0, double %1)
double t3(double a, double b) {
  return __builtin_HEXAGON_F2_dfmax(a, b);
}

// CHECK-LABEL: @t4
// CHECK: call double @llvm.hexagon.F2.dfmin(double %0, double %1)
double t4(double a, double b) {
  return __builtin_HEXAGON_F2_dfmin(a, b);
}

// CHECK-LABEL: @t5
// CHECK: call double @llvm.hexagon.F2.dfmpyfix(double %0, double %1)
double t5(double a, double b) {
  return __builtin_HEXAGON_F2_dfmpyfix(a, b);
}

// CHECK-LABEL: @t6
// CHECK: call double @llvm.hexagon.F2.dfmpyll(double %0, double %1)
double t6(double a, double b) {
  return __builtin_HEXAGON_F2_dfmpyll(a, b);
}

// CHECK-LABEL: @t7
// CHECK: call i64 @llvm.hexagon.M7.vdmpy(i64 %0, i64 %1)
long long t7(long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_vdmpy(rss, rtt);
}

// CHECK-LABEL: @t8
// CHECK: call i64 @llvm.hexagon.M7.vdmpy.acc(i64 %0, i64 %1, i64 %2)
long long t8(long long rxx, long long rss, long long rtt) {
  return __builtin_HEXAGON_M7_vdmpy_acc(rxx, rss, rtt);
}

