// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_cache() {
  // CHECK: cir.func{{.*}} @acc_cache()

  int iArr[10];
  // CHECK-NEXT: %[[IARR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["iArr"]
  float fArr[10];
  // CHECK-NEXT: %[[FARR:.*]] = cir.alloca !cir.array<!cir.float x 10>, !cir.ptr<!cir.array<!cir.float x 10>>, ["fArr"]

#pragma acc cache(iArr[1], fArr[1:5])
  // This does nothing, as it is not in a loop.

#pragma acc parallel
  {
#pragma acc cache(iArr[1], fArr[1:5])
  // This does nothing, as it is not in a loop.
  }
  // CHECK-NEXT: acc.parallel {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc loop
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j) {
#pragma acc cache(iArr[1], fArr[1:5])
    }
  }
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE1:.*]] = acc.cache varPtr(%[[IARR]] : !cir.ptr<!cir.array<!s32i x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 10>> {name = "iArr[1]", structured = false}
  //
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE:.*]] = cir.const #cir.int<5>
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[FIVE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE2:.*]] = acc.cache varPtr(%[[FARR]] : !cir.ptr<!cir.array<!cir.float x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 10>> {name = "fArr[1:5]", structured = false}
  //
  // CHECK-NEXT: acc.loop cache(%[[CACHE1]], %[[CACHE2]] : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.array<!cir.float x 10>>) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]}

#pragma acc loop
  for(int i = 0; i < 5; ++i) {
#pragma acc cache(iArr[1], fArr[1:5])
  }
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE1:.*]] = acc.cache varPtr(%[[IARR]] : !cir.ptr<!cir.array<!s32i x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 10>> {name = "iArr[1]", structured = false}
  //
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE:.*]] = cir.const #cir.int<5>
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[FIVE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE2:.*]] = acc.cache varPtr(%[[FARR]] : !cir.ptr<!cir.array<!cir.float x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 10>> {name = "fArr[1:5]", structured = false}
  //
  // CHECK-NEXT: acc.loop cache(%[[CACHE1]], %[[CACHE2]] : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.array<!cir.float x 10>>) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]}

#pragma acc parallel loop
  for(int i = 0; i < 5; ++i) {
#pragma acc cache(iArr[1], fArr[1:5])
  }
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE1:.*]] = acc.cache varPtr(%[[IARR]] : !cir.ptr<!cir.array<!s32i x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 10>> {name = "iArr[1]", structured = false}
  //
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE:.*]] = cir.const #cir.int<5>
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[FIVE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE2:.*]] = acc.cache varPtr(%[[FARR]] : !cir.ptr<!cir.array<!cir.float x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 10>> {name = "fArr[1:5]", structured = false}
  //
  // CHECK-NEXT: acc.loop combined(parallel) cache(%[[CACHE1]], %[[CACHE2]] : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.array<!cir.float x 10>>) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]}
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop
  for(int i = 0; i < 5; ++i) {
    int localArr[5];
    // The first term here isn't lowered, because it references data inside of the 'loop'.
#pragma acc cache(localArr[i], iArr[1], fArr[1:5])
  }
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE1:.*]] = acc.cache varPtr(%[[IARR]] : !cir.ptr<!cir.array<!s32i x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 10>> {name = "iArr[1]", structured = false}
  //
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE:.*]] = cir.const #cir.int<5>
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[FIVE_CAST]] : si32) stride(%[[ONE_CONST]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[CACHE2:.*]] = acc.cache varPtr(%[[FARR]] : !cir.ptr<!cir.array<!cir.float x 10>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 10>> {name = "fArr[1:5]", structured = false}
  //
  // CHECK-NEXT: acc.loop combined(parallel) cache(%[[CACHE1]], %[[CACHE2]] : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.array<!cir.float x 10>>) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]}
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
