// REQUIRES: hexagon-registered-target
// The -emit-llvm run checks that ordinary pair memory operations are emitted,
// the -stop-after run checks their lowering, and the last run checks that the
// full backend accepts them.
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -O0 -S %s -o - -mllvm -stop-after=hexagon-lower-hlx-pair-allocas | FileCheck %s --check-prefix=LOWER
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -O0 -S %s -o /dev/null

#include <hexagon_types.h>
#include <hlx_hexagon_protos.h>

HLX_VectorPair copy_hlx_vector_pair(HLX_VectorPair input) {
  HLX_VectorPair copy = input;
  return copy;
}

// IR-LABEL: define{{.*}} <1024 x i32> @copy_hlx_vector_pair(
// IR-SAME: %[[ARG:.*]])
// IR: %[[INPUT_ADDR:.*]] = alloca <1024 x i32>, align 2048
// IR: %[[COPY_ADDR:.*]] = alloca <1024 x i32>, align 2048
// IR: store <1024 x i32> %[[ARG]], ptr %[[INPUT_ADDR]], align 2048
// IR: %[[INPUT:.*]] = load <1024 x i32>, ptr %[[INPUT_ADDR]], align 2048
// IR: store <1024 x i32> %[[INPUT]], ptr %[[COPY_ADDR]], align 2048
// IR: %[[COPY:.*]] = load <1024 x i32>, ptr %[[COPY_ADDR]], align 2048
// IR: ret <1024 x i32> %[[COPY]]
// IR-NOT: @llvm.hexagon.LX.

// LOWER-LABEL: define{{.*}} <1024 x i32> @copy_hlx_vector_pair(
// LOWER-SAME: %[[ARG:.*]])
// LOWER-NOT: load <1024 x i32>
// LOWER-NOT: store <1024 x i32>
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %[[INPUT_ADDR:.*]], <1024 x i32> %[[ARG]])
// LOWER: %[[INPUT:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %[[INPUT_ADDR]])
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %[[COPY_ADDR:.*]], <1024 x i32> %[[INPUT]])
// LOWER: %[[COPY:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %[[COPY_ADDR]])
// LOWER: ret <1024 x i32> %[[COPY]]
