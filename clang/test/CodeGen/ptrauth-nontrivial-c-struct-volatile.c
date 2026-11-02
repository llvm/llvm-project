// RUN: %clang_cc1 -triple arm64-apple-ios -fblocks -fptrauth-calls -fptrauth-returns -fptrauth-intrinsics -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#define AQ __ptrauth(1, 1, 50)

struct HasScalarField {
  void (* AQ fp)(void);
  int f3;
};

// CHECK-LABEL: define{{.*}} void @copyScalar(
// CHECK: call void @[[COPY_ASSIGNMENT_SCALAR:__copy_assignment[a-zA-Z0-9_]*]](
void copyScalar(volatile struct HasScalarField *dst,
                 volatile struct HasScalarField *src) {
  *dst = *src;
}

// CHECK: define{{.*}} void @[[COPY_ASSIGNMENT_SCALAR]](
// CHECK: %[[DST:.*]] = getelementptr inbounds nuw %struct.HasScalarField, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[SRC:.*]] = getelementptr inbounds nuw %struct.HasScalarField, ptr %{{.*}}, i32 0, i32 1
// CHECK: load volatile i32, ptr %[[SRC]], align 8, !tbaa ![[TBAA_SCALAR:[0-9]+]]{{$}}
// CHECK: store volatile i32 %{{.*}}, ptr %[[DST]], align 8, !tbaa ![[TBAA_SCALAR]]{{$}}
// CHECK: ret void

struct Inner {
  int a;
  int b;
};

struct HasAggregateField {
  void (* AQ fp)(void);
  struct Inner f3;
};

// CHECK-LABEL: define{{.*}} void @copyAggregate(
// CHECK: call void @[[COPY_ASSIGNMENT_AGGREGATE:__copy_assignment[a-zA-Z0-9_]*]](
void copyAggregate(volatile struct HasAggregateField *dst,
                    volatile struct HasAggregateField *src) {
  *dst = *src;
}

// CHECK: define{{.*}} void @[[COPY_ASSIGNMENT_AGGREGATE]](
// CHECK: %[[DST:.*]] = getelementptr inbounds nuw %struct.HasAggregateField, ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[SRC:.*]] = getelementptr inbounds nuw %struct.HasAggregateField, ptr %{{.*}}, i32 0, i32 1
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[DST]], ptr align 8 %[[SRC]], i64 8, i1 true), !tbaa.struct ![[TBAA_STRUCT:[0-9]+]]{{$}}
// CHECK: ret void

struct HasComplexField {
  void (* AQ fp)(void);
  _Complex double f3;
};

// CHECK-LABEL: define{{.*}} void @copyComplex(
// CHECK: call void @[[COPY_ASSIGNMENT_COMPLEX:__copy_assignment[a-zA-Z0-9_]*]](
void copyComplex(volatile struct HasComplexField *dst,
                  volatile struct HasComplexField *src) {
  *dst = *src;
}

// _Complex fields aren't given any TBAA at all.
// CHECK: define{{.*}} void @[[COPY_ASSIGNMENT_COMPLEX]](
// CHECK: %[[SRC_REAL:.*]] = load volatile double, ptr %{{.*}}, align 8{{$}}
// CHECK: %[[SRC_IMAG:.*]] = load volatile double, ptr %{{.*}}, align 8{{$}}
// CHECK: store volatile double %[[SRC_REAL]], ptr %{{.*}}, align 8{{$}}
// CHECK: store volatile double %[[SRC_IMAG]], ptr %{{.*}}, align 8{{$}}
// CHECK: ret void

// CHECK: ![[TBAA_INT:[0-9]+]] = !{!"int", ![[TBAA_CHAR:[0-9]+]], i64 0}
// CHECK: ![[TBAA_CHAR]] = !{!"omnipotent char", ![[TBAA_DOMAIN:[0-9]+]], i64 0}
// CHECK: ![[TBAA_DOMAIN]] = !{!"Simple C/C++ TBAA"}
// CHECK: ![[TBAA_PTR:[0-9]+]] = !{!"any pointer", ![[TBAA_CHAR]], i64 0}
// CHECK: ![[TBAA_SCALAR]] = !{![[TBAA_SCALAR_BASE:[0-9]+]], ![[TBAA_INT]], i64 8}
// CHECK: ![[TBAA_SCALAR_BASE]] = !{!"HasScalarField", ![[TBAA_PTR]], i64 0, ![[TBAA_INT]], i64 8}

// CHECK: ![[TBAA_STRUCT]] = !{i64 0, i64 4, ![[TBAA_STRUCT_TAG:[0-9]+]], i64 4, i64 4, ![[TBAA_STRUCT_TAG]]}
// CHECK: ![[TBAA_STRUCT_TAG]] = !{![[TBAA_INT]], ![[TBAA_INT]], i64 0}
