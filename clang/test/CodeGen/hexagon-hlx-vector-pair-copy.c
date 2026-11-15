// REQUIRES: hexagon-registered-target
// The -emit-llvm run checks that ordinary pair memory operations are emitted,
// the -stop-after run checks their lowering, and the last run checks that the
// full backend accepts them.
// RUN: %clang --target=hexagon -mv85 -mhlx -O0 -S %s -o - -mllvm -stop-after=hexagon-lower-hlx-pair-allocas | FileCheck %s --check-prefix=LOWER
// RUN: %clang --target=hexagon -mv85 -mhlx -O0 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=IR
// RUN: %clang --target=hexagon -mv85 -mhlx -O0 -S %s -o /dev/null

#include <hexagon_types.h>
#include <hlx_hexagon_protos.h>

HLX_VectorPair copy_hlx_vector_pair(HLX_VectorPair input) {
  HLX_VectorPair copy = input;
  return copy;
}

// IR-LABEL: define{{.*}} <1024 x i32> @copy_hlx_vector_pair
// IR: %input.addr = alloca <1024 x i32>, align 2048
// IR: %copy = alloca <1024 x i32>, align 2048
// IR: store <1024 x i32> %input, ptr %input.addr, align 2048
// IR: %[[INPUT:.*]] = load <1024 x i32>, ptr %input.addr, align 2048
// IR: store <1024 x i32> %[[INPUT]], ptr %copy, align 2048
// IR: %[[COPY:.*]] = load <1024 x i32>, ptr %copy, align 2048
// IR: ret <1024 x i32> %[[COPY]]
// IR-NOT: @llvm.hexagon.LX.

// LOWER-LABEL: define{{.*}} <1024 x i32> @copy_hlx_vector_pair
// LOWER-NOT: load <1024 x i32>
// LOWER-NOT: store <1024 x i32>
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %input.addr, <1024 x i32> %input)
// LOWER: %[[INPUT:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %input.addr)
// LOWER: call void @llvm.hexagon.LX.SS16b.new(ptr %copy, <1024 x i32> %[[INPUT]])
// LOWER: %[[COPY:.*]] = call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %copy)
// LOWER: ret <1024 x i32> %[[COPY]]
