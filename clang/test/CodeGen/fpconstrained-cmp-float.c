// RUN: %clang_cc1 -ffp-exception-behavior=ignore -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FCMP
// RUN: %clang_cc1 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=EXCEPT
// RUN: %clang_cc1 -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=ignore -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=IGNORE
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FCMP
// RUN: %clang_cc1 -frounding-math -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=MAYTRAP-RND

_Bool QuietEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp oeq float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oeq") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 == f2;

  // CHECK: ret
}

_Bool QuietNotEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietNotEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp une float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"une") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 != f2;

  // CHECK: ret
}

_Bool SignalingLess(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLess(float noundef %f1, float noundef %f2)

  // FCMP: fcmp olt float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 < f2;

  // CHECK: ret
}

_Bool SignalingLessEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingLessEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp ole float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 <= f2;

  // CHECK: ret
}

_Bool SignalingGreater(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreater(float noundef %f1, float noundef %f2)

  // FCMP: fcmp ogt float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 > f2;

  // CHECK: ret
}

_Bool SignalingGreaterEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @SignalingGreaterEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp oge float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return f1 >= f2;

  // CHECK: ret
}

_Bool QuietLess(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLess(float noundef %f1, float noundef %f2)

  // FCMP: fcmp olt float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"olt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isless(f1, f2);

  // CHECK: ret
}

_Bool QuietLessEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp ole float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ole") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_islessequal(f1, f2);

  // CHECK: ret
}

_Bool QuietGreater(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreater(float noundef %f1, float noundef %f2)

  // FCMP: fcmp ogt float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"ogt") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietGreaterEqual(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietGreaterEqual(float noundef %f1, float noundef %f2)

  // FCMP: fcmp oge float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"oge") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isgreaterequal(f1, f2);

  // CHECK: ret
}

_Bool QuietLessGreater(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietLessGreater(float noundef %f1, float noundef %f2)

  // FCMP: fcmp one float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"one") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"one") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"one") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"one") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_islessgreater(f1, f2);

  // CHECK: ret
}

_Bool QuietUnordered(float f1, float f2) {
  // CHECK-LABEL: define {{.*}}i1 @QuietUnordered(float noundef %f1, float noundef %f2)

  // FCMP: fcmp uno float %{{.*}}, %{{.*}}
  // IGNORE: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"uno") #{{.*}} [ "fp.except"(metadata !"ignore") ]
  // EXCEPT: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"uno") #{{.*}} [ "fp.control"(metadata !"rte") ]
  // MAYTRAP: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"uno") #{{.*}} [ "fp.control"(metadata !"rte"), "fp.except"(metadata !"maytrap") ]
  // MAYTRAP-RND: call i1 @llvm.fcmp.f32(float %{{.*}}, float %{{.*}}, metadata !"uno") #{{.*}} [ "fp.except"(metadata !"maytrap") ]
  return __builtin_isunordered(f1, f2);

  // CHECK: ret
}
