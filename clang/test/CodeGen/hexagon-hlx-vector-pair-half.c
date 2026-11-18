// REQUIRES: hexagon-registered-target
// RUN: %clang --target=hexagon -mv85 -mhlx -O0 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang --target=hexagon -mv85 -mhlx -O0 -S %s -o - -mllvm -stop-after=hexagon-lower-hlx-pair-allocas | FileCheck %s --check-prefix=LOWER

#include <hexagon_types.h>
#include <hlx_hexagon_protos.h>

HLX_Vector get_half_from_pair(HLX_VectorPair input) {
  HLX_VectorPair pair = input;
  HLX_Vector half = *(HLX_Vector *)&pair;
  return half;
}

// IR-LABEL: define{{.*}} <1024 x i16> @get_half_from_pair
// IR: %pair = alloca <1024 x i32>, align 2048
// IR: store <1024 x i32> %input, ptr %input.addr, align 2048
// IR: %[[PAIR_VALUE:.*]] = load <1024 x i32>, ptr %input.addr, align 2048
// IR: store <1024 x i32> %[[PAIR_VALUE]], ptr %pair, align 2048
// IR: %[[HALF:.*]] = load <1024 x i16>, ptr %pair, align 2048
// IR-NOT: @llvm.hexagon.LX.

// LOWER-LABEL: define{{.*}} <1024 x i16> @get_half_from_pair
// LOWER-NOT: load <1024 x i32>
// LOWER-NOT: store <1024 x i32>
// LOWER: call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %input.addr)
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %pair, <1024 x i32>
// LOWER: %{{.*}} = load <1024 x i16>, ptr %pair, align 2048
