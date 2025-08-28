// RUN: %clang -Xclang -disable-llvm-passes -S -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

// CHECK-LABEL: define{{.*}}void @check(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
// CHECK:    [[BS:%.*]] = alloca ptr
// CHECK:    [[TMP0:%.*]] = call ptr @llvm.ripple.block.setshape.i{{(32|64)}}(i{{(32|64)}} 0, i{{(32|64)}} 32, i{{(32|64)}} 4, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1, i{{(32|64)}} 1)
// CHECK-NEXT:    store ptr [[TMP0]], ptr [[BS]]
// CHECK:    [[TMP1:%.*]] = load ptr, ptr [[BS]]
// CHECK-NEXT:    [[TMP2:%.*]] = call i{{(32|64)}} @llvm.ripple.block.index.i{{(32|64)}}(ptr [[TMP1]], i{{(32|64)}} 0)
// CHECK:    [[TMP3:%.*]] = load ptr, ptr [[BS]]
// CHECK-NEXT:    [[TMP4:%.*]] = call i{{(32|64)}} @llvm.ripple.block.index.i{{(32|64)}}(ptr [[TMP3]], i{{(32|64)}} 2)
// CHECK:    ret void
//
void check() {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4);
  int idx_x = ripple_id(BS, 0);
  int idx_y = ripple_id(BS, 2);
}
