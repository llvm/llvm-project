// RUN: %clang_cc1 -triple aarch64_be-apple-darwin -emit-llvm -o - -O1 %s | \
// RUN:     FileCheck -check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple aarch64-apple-darwin -emit-llvm -o - -O1 %s | \
// RUN:     FileCheck -check-prefixes=CHECK %s
//
// Check that TBAA metadata for structs containing bitfields is
// consistent between big and little endian layouts.

struct NamedBitfields {
  int f1 : 8;
  int f2 : 8;
  unsigned f3 : 1;
  unsigned f4 : 15;
  int f5;
  double f6;
};

// CHECK-LABEL: _Z4copyP14NamedBitfieldsS0_
// CHECK-SAME: ptr noundef writeonly captures(none) initializes((0, 16)) [[A1:%.*]], ptr noundef readonly captures(none) [[A2:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) [[A1]], ptr noundef nonnull align 8 dereferenceable(16) [[A2]], i64 16, i1 false), !tbaa.struct [[TBAA_STRUCT6:![0-9]+]]
// CHECK-NEXT:    ret void
//
void copy(NamedBitfields *a1, NamedBitfields *a2) {
  *a1 = *a2;
}

// CHECK: [[META2:![0-9]+]] = !{[[META3:![0-9]+]], [[META3]], i64 0}
// CHECK: [[META3]] = !{!"int", [[META4:![0-9]+]], i64 0}
// CHECK: [[META4]] = !{!"omnipotent char", [[META5:![0-9]+]], i64 0}
// CHECK: [[META5]] = !{!"Simple C++ TBAA"}
// CHECK: [[TBAA_STRUCT6]] = !{i64 0, i64 4, [[META7:![0-9]+]], i64 4, i64 4, [[META2:![0-9]+]], i64 8, i64 8, [[META8:![0-9]+]]}
// CHECK: [[META7]] = !{[[META4:![0-9]+]], [[META4]], i64 0}
// CHECK: [[META8]] = !{[[META9:![0-9]+]], [[META9]], i64 0}
// CHECK: [[META9]] = !{!"double", [[META4]], i64 0}
