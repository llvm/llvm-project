// RUN: %clang_cc1 -ffp-exception-behavior=ignore -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FCMP
// RUN: %clang_cc1 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=EXCEPT
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=ignore -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=IGNORE
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=EXCEPT
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP

_Bool QuietEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietEqual(double %f1, double %f2)

  // FCMP: fcmp oeq double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq", metadata !"fpexcept.maytrap")
  return f1 == f2;

  // CHECK: ret
}

_Bool QuietNotEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietNotEqual(double %f1, double %f2)

  // FCMP: fcmp une double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une", metadata !"fpexcept.maytrap")
  return f1 != f2;

  // CHECK: ret
}

_Bool SignalingLess(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLess(double %f1, double %f2)

  // FCMP: fcmp olt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.maytrap")
  return f1 < f2;

  // CHECK: ret
}

_Bool SignalingLessEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLessEqual(double %f1, double %f2)

  // FCMP: fcmp ole double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.maytrap")
  return f1 <= f2;

  // CHECK: ret
}

_Bool SignalingGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreater(double %f1, double %f2)

  // FCMP: fcmp ogt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.maytrap")
  return f1 > f2;

  // CHECK: ret
}

_Bool SignalingGreaterEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreaterEqual(double %f1, double %f2)

  // FCMP: fcmp oge double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmps.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.maytrap")
  return f1 >= f2;

  // CHECK: ret
}

_Bool QuietLess(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLess(double %f1, double %f2)

  // FCMP: fcmp olt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt", metadata !"fpexcept.maytrap")
  return __builtin_isless(f1, f2);

  // CHECK: ret
}

_Bool QuietLessEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessEqual(double %f1, double %f2)

  // FCMP: fcmp ole double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole", metadata !"fpexcept.maytrap")
  return __builtin_islessequal(f1, f2);

  // CHECK: ret
}

_Bool QuietGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreater(double %f1, double %f2)

  // FCMP: fcmp ogt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt", metadata !"fpexcept.maytrap")
  return __builtin_isgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietGreaterEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreaterEqual(double %f1, double %f2)

  // FCMP: fcmp oge double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge", metadata !"fpexcept.maytrap")
  return __builtin_isgreaterequal(f1, f2);

  // CHECK: ret
}

_Bool QuietLessGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessGreater(double %f1, double %f2)

  // FCMP: fcmp one double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one", metadata !"fpexcept.maytrap")
  return __builtin_islessgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietUnordered(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietUnordered(double %f1, double %f2)

  // FCMP: fcmp uno double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno", metadata !"fpexcept.ignore")
  // EXCEPT: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // MAYTRAP: call i1 @llvm.experimental.constrained.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno", metadata !"fpexcept.maytrap")
  return __builtin_isunordered(f1, f2);

  // CHECK: ret
}

