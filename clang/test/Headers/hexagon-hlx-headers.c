// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv85 -triple hexagon-unknown-elf \
// RUN:   -target-feature +hlx -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv85 -triple hexagon-unknown-elf -x c++ \
// RUN:   -target-feature +hlx -emit-llvm %s -o - | FileCheck %s

#include <hexagon_protos.h>
#include <hexagon_types.h>

// CHECK-LABEL: test_hlx_protos
void test_hlx_protos(void *a, void *b, void *c) {
  // CHECK: call <1024 x i16> @llvm.hexagon.LX.L16b.tmp(ptr
  HLX_Vector x = Q6_L_tmp_mem_A(a);
  // CHECK: call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr
  HLX_VectorPair y = Q6_LL_tmp_mem_A(b);
  // CHECK: call <1024 x i16> @llvm.hexagon.LX.max.uh(<1024 x i16>
  x = Q6_Luh_max_LuhLuh(x, x);
  // CHECK: call void @llvm.hexagon.LX.S16b.new(ptr {{.*}}, <1024 x i16>
  Q6_mem_new_AL(c, x);
  // CHECK: call void @llvm.hexagon.LX.SS16b.new(ptr {{.*}}, <1024 x i32>
  Q6_mem_new_ALL(c, y);
}
