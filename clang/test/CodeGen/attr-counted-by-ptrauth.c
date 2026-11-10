// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsanitize=array-bounds -fsanitize-trap=array-bounds -emit-llvm -o - %s | FileCheck %s

// A '__ptrauth'-qualified 'counted_by' pointer must still be authenticated
// before it's indexed when -fsanitize=array-bounds shares the field's l-value
// with the bounds check.

struct pa {
  int count;
  int *__ptrauth(1, 1, 42) buf __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_index(
// CHECK:       [[BUF:%.*]] = getelementptr inbounds nuw %struct.pa, ptr {{%.*}}, i32 0, i32 1
// CHECK:       [[SIGNED:%.*]] = load ptr, ptr [[BUF]]
// CHECK:       [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 42)
// CHECK:       [[INT:%.*]] = ptrtoint ptr [[SIGNED]] to i64
// CHECK:       [[AUTH:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[INT]], i32 1, i64 [[DISC]])
// CHECK:       [[PTR:%.*]] = inttoptr i64 [[AUTH]] to ptr
// CHECK:       [[PHI:%.*]] = phi ptr [ null, {{.*}} ], [ [[PTR]], {{.*}} ]
// CHECK:       getelementptr inbounds i32, ptr [[PHI]], i64
// CHECK:       call void @llvm.ubsantrap(
int test_index(struct pa *p, int i) {
  return p->buf[i];
}
