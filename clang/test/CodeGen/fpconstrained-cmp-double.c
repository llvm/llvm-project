// RUN: %clang_cc1 -ffp-exception-behavior=ignore -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FCMP
// RUN: %clang_cc1 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=EXCEPT
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=ignore -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=IGNORE
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FCMP
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP-RND

_Bool QuietEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp oeq double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 == f2;

  // CHECK: ret
}

_Bool QuietNotEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietNotEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp une double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"une") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 != f2;

  // CHECK: ret
}

_Bool SignalingLess(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLess(double noundef %f1, double noundef %f2)

  // FCMP: fcmp olt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 < f2;

  // CHECK: ret
}

_Bool SignalingLessEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLessEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp ole double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 <= f2;

  // CHECK: ret
}

_Bool SignalingGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreater(double noundef %f1, double noundef %f2)

  // FCMP: fcmp ogt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 > f2;

  // CHECK: ret
}

_Bool SignalingGreaterEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreaterEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp oge double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 >= f2;

  // CHECK: ret
}

_Bool QuietLess(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLess(double noundef %f1, double noundef %f2)

  // FCMP: fcmp olt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isless(f1, f2);

  // CHECK: ret
}

_Bool QuietLessEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp ole double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_islessequal(f1, f2);

  // CHECK: ret
}

_Bool QuietGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreater(double noundef %f1, double noundef %f2)

  // FCMP: fcmp ogt double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietGreaterEqual(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreaterEqual(double noundef %f1, double noundef %f2)

  // FCMP: fcmp oge double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isgreaterequal(f1, f2);

  // CHECK: ret
}

_Bool QuietLessGreater(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessGreater(double noundef %f1, double noundef %f2)

  // FCMP: fcmp one double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"one") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_islessgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietUnordered(double f1, double f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietUnordered(double noundef %f1, double noundef %f2)

  // FCMP: fcmp uno double %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f64(double %{{.*}}, double %{{.*}}, metadata !"uno") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isunordered(f1, f2);

  // CHECK: ret
}

