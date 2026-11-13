// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple hexagon -target-cpu hexagonv85 -target-feature +hlx -O0 -S %s -o - -mllvm -stop-after=hexagon-lower-hlx-pair-allocas | FileCheck %s --check-prefix=LOWER

#include <hexagon_types.h>
#include <hlx_hexagon_protos.h>

HLX_Vector get_half_from_pair(HLX_VectorPair input) {
  HLX_VectorPair pair = input;
  HLX_Vector half = *(HLX_Vector *)&pair;
  return half;
}

// IR-LABEL: define{{.*}} <1024 x i16> @get_half_from_pair(
// IR-SAME: %[[ARG:.*]])
// IR: %[[INPUT_ADDR:.*]] = alloca <1024 x i32>, align 2048
// IR: %[[PAIR_ADDR:.*]] = alloca <1024 x i32>, align 2048
// IR: store <1024 x i32> %[[ARG]], ptr %[[INPUT_ADDR]], align 2048
// IR: %[[PAIR:.*]] = load <1024 x i32>, ptr %[[INPUT_ADDR]], align 2048
// IR: store <1024 x i32> %[[PAIR]], ptr %[[PAIR_ADDR]], align 2048
// IR: load <1024 x i16>, ptr %[[PAIR_ADDR]], align 2048
// IR-NOT: @llvm.hexagon.LX.

// LOWER-LABEL: define{{.*}} <1024 x i16> @get_half_from_pair(
// LOWER-SAME: %[[ARG:.*]])
// LOWER-NOT: load <1024 x i32>
// LOWER-NOT: store <1024 x i32>
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %[[INPUT_ADDR:.*]], <1024 x i32> %[[ARG]])
// LOWER: %[[PAIR:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %[[INPUT_ADDR]])
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %[[PAIR_ADDR:.*]], <1024 x i32> %[[PAIR]])
// LOWER: load <1024 x i16>, ptr %[[PAIR_ADDR]], align 2048
