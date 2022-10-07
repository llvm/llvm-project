// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -ffixed-point %s -emit-llvm -o - | FileCheck %s  -check-prefixes=CHECK
//
// Check that we generate correct TBAA metadata for fixed-point types.

void sfract(unsigned short _Fract *p, short _Fract *q,
            unsigned _Sat short _Fract *r, _Sat short _Fract *s) {
  // CHECK-LABEL: define{{.*}} void @sfract
  // CHECK: store i8 -128, ptr %p, align 1, !tbaa [[TAG_sf:!.*]]
  // CHECK: store i8 -64, ptr %q, align 1, !tbaa [[TAG_sf]]
  // CHECK: store i8 -128, ptr %r, align 1, !tbaa [[TAG_sat_sf:!.*]]
  // CHECK: store i8 -64, ptr %s, align 1, !tbaa [[TAG_sat_sf]]
  *p = 0.5hur;
  *q = -0.5hr;
  *r = 0.5hur;
  *s = -0.5hr;
}

void fract(unsigned _Fract *p, _Fract *q,
           unsigned _Sat _Fract *r, _Sat _Fract *s) {
  // CHECK-LABEL: define{{.*}} void @fract
  // CHECK: store i16 -32768, ptr %p, align 2, !tbaa [[TAG_f:!.*]]
  // CHECK: store i16 -16384, ptr %q, align 2, !tbaa [[TAG_f]]
  // CHECK: store i16 -32768, ptr %r, align 2, !tbaa [[TAG_sat_f:!.*]]
  // CHECK: store i16 -16384, ptr %s, align 2, !tbaa [[TAG_sat_f]]
  *p = 0.5ur;
  *q = -0.5r;
  *r = 0.5ur;
  *s = -0.5r;
}

void lfract(unsigned long _Fract *p, long _Fract *q,
            unsigned _Sat long _Fract *r, _Sat long _Fract *s) {
  // CHECK-LABEL: define{{.*}} void @lfract
  // CHECK: store i32 -2147483648, ptr %p, align 4, !tbaa [[TAG_lf:!.*]]
  // CHECK: store i32 -1073741824, ptr %q, align 4, !tbaa [[TAG_lf]]
  // CHECK: store i32 -2147483648, ptr %r, align 4, !tbaa [[TAG_sat_lf:!.*]]
  // CHECK: store i32 -1073741824, ptr %s, align 4, !tbaa [[TAG_sat_lf]]
  *p = 0.5ulr;
  *q = -0.5lr;
  *r = 0.5ulr;
  *s = -0.5lr;
}

void saccum(unsigned short _Accum *p, short _Accum *q,
            unsigned _Sat short _Accum *r, _Sat short _Accum *s) {
  // CHECK-LABEL: define{{.*}} void @saccum
  // CHECK: store i16 128, ptr %p, align 2, !tbaa [[TAG_sk:!.*]]
  // CHECK: store i16 -64, ptr %q, align 2, !tbaa [[TAG_sk]]
  // CHECK: store i16 128, ptr %r, align 2, !tbaa [[TAG_sat_sk:!.*]]
  // CHECK: store i16 -64, ptr %s, align 2, !tbaa [[TAG_sat_sk]]
  *p = 0.5huk;
  *q = -0.5hk;
  *r = 0.5huk;
  *s = -0.5hk;
}

void accum(unsigned _Accum *p, _Accum *q,
           unsigned _Sat _Accum *r, _Sat _Accum *s) {
  // CHECK-LABEL: define{{.*}} void @accum
  // CHECK: store i32 32768, ptr %p, align 4, !tbaa [[TAG_k:!.*]]
  // CHECK: store i32 -16384, ptr %q, align 4, !tbaa [[TAG_k]]
  // CHECK: store i32 32768, ptr %r, align 4, !tbaa [[TAG_sat_k:!.*]]
  // CHECK: store i32 -16384, ptr %s, align 4, !tbaa [[TAG_sat_k]]
  *p = 0.5uk;
  *q = -0.5k;
  *r = 0.5uk;
  *s = -0.5k;
}

void laccum(unsigned long _Accum *p, long _Accum *q,
            unsigned _Sat long _Accum *r, _Sat long _Accum *s) {
  // CHECK-LABEL: define{{.*}} void @laccum
  // CHECK: store i64 2147483648, ptr %p, align 8, !tbaa [[TAG_lk:!.*]]
  // CHECK: store i64 -1073741824, ptr %q, align 8, !tbaa [[TAG_lk]]
  // CHECK: store i64 2147483648, ptr %r, align 8, !tbaa [[TAG_sat_lk:!.*]]
  // CHECK: store i64 -1073741824, ptr %s, align 8, !tbaa [[TAG_sat_lk]]
  *p = 0.5ulk;
  *q = -0.5lk;
  *r = 0.5ulk;
  *s = -0.5lk;
}

// CHECK-DAG: [[TAG_sf]] = !{[[TYPE_sf:!.*]], [[TYPE_sf]], i64 0}
// CHECK-DAG: [[TYPE_sf]] = !{!"short _Fract"
// CHECK-DAG: [[TAG_f]] = !{[[TYPE_f:!.*]], [[TYPE_f]], i64 0}
// CHECK-DAG: [[TYPE_f]] = !{!"_Fract"
// CHECK-DAG: [[TAG_lf]] = !{[[TYPE_lf:!.*]], [[TYPE_lf]], i64 0}
// CHECK-DAG: [[TYPE_lf]] = !{!"long _Fract"

// CHECK-DAG: [[TAG_sat_sf]] = !{[[TYPE_sat_sf:!.*]], [[TYPE_sat_sf]], i64 0}
// CHECK-DAG: [[TYPE_sat_sf]] = !{!"_Sat short _Fract"
// CHECK-DAG: [[TAG_sat_f]] = !{[[TYPE_sat_f:!.*]], [[TYPE_sat_f]], i64 0}
// CHECK-DAG: [[TYPE_sat_f]] = !{!"_Sat _Fract"
// CHECK-DAG: [[TAG_sat_lf]] = !{[[TYPE_sat_lf:!.*]], [[TYPE_sat_lf]], i64 0}
// CHECK-DAG: [[TYPE_sat_lf]] = !{!"_Sat long _Fract"

// CHECK-DAG: [[TAG_sk]] = !{[[TYPE_sk:!.*]], [[TYPE_sk]], i64 0}
// CHECK-DAG: [[TYPE_sk]] = !{!"short _Accum"
// CHECK-DAG: [[TAG_k]] = !{[[TYPE_k:!.*]], [[TYPE_k]], i64 0}
// CHECK-DAG: [[TYPE_k]] = !{!"_Accum"
// CHECK-DAG: [[TAG_lk]] = !{[[TYPE_lk:!.*]], [[TYPE_lk]], i64 0}
// CHECK-DAG: [[TYPE_lk]] = !{!"long _Accum"

// CHECK-DAG: [[TAG_sat_sk]] = !{[[TYPE_sat_sk:!.*]], [[TYPE_sat_sk]], i64 0}
// CHECK-DAG: [[TYPE_sat_sk]] = !{!"_Sat short _Accum"
// CHECK-DAG: [[TAG_sat_k]] = !{[[TYPE_sat_k:!.*]], [[TYPE_sat_k]], i64 0}
// CHECK-DAG: [[TYPE_sat_k]] = !{!"_Sat _Accum"
// CHECK-DAG: [[TAG_sat_lk]] = !{[[TYPE_sat_lk:!.*]], [[TYPE_sat_lk]], i64 0}
// CHECK-DAG: [[TYPE_sat_lk]] = !{!"_Sat long _Accum"
