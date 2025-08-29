// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

// This encounters NYI errors because of a non-ignored copy of an aggregate.
// When that is fixed, the 'not' should be removed from the RUN line above.

struct NoCopyConstruct {};

// CHECK: acc.firstprivate.recipe @firstprivatization__ZTSA5_15NoCopyConstruct : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: %[[DECAY_TO:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>), !cir.ptr<!rec_NoCopyConstruct>
// TODO: OpenACC: cir.copy isn't implemented correctly yet, so this doesn't actually do any initialization.
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[ONE]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[TWO]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[THREE]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!rec_NoCopyConstruct>, %[[FOUR]] : !s64i), !cir.ptr<!rec_NoCopyConstruct>
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!cir.float>, %[[ZERO]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_DECAY]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[ONE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[ONE_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[TWO]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[TWO_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[THREE]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[THREE_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!cir.float>, %[[FOUR]] : !s64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.float x 5>>), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!cir.float>, %[[FOUR_2]] : !u64i), !cir.ptr<!cir.float>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: %[[TO_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_TO]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT: %[[FROM_DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[FROM_DECAY]] : !cir.ptr<!s32i>, %[[ZERO]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_DECAY]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[ONE_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[TWO:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[TWO]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_2:.*]] = cir.const #cir.int<2>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[TWO_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[THREE:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[THREE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_2:.*]] = cir.const #cir.int<3>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[THREE_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
//
// CHECK-NEXT: %[[FOUR:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[TO_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_TO]] : !cir.ptr<!s32i>, %[[FOUR]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_2:.*]] = cir.const #cir.int<4>
// CHECK-NEXT: %[[DECAY_FROM:.*]] =  cir.cast(array_to_ptrdecay, %[[ARG_FROM]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_OFFSET:.*]] = cir.ptr_stride(%[[DECAY_FROM]] : !cir.ptr<!s32i>, %[[FOUR_2]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[FROM_OFFSET]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[TO_OFFSET]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS15NoCopyConstruct : !cir.ptr<!rec_NoCopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
// TODO: OpenACC: This should emit a copy, but cir.copy isn't implemented yet.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSf : !cir.ptr<!cir.float> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.float> {{.*}}):
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[ARG_FROM]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[ARG_TO]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.firstprivate.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!s32i> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: %[[FROM_LOAD:.*]] = cir.load {{.*}}%[[ARG_FROM]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: cir.store {{.*}} %[[FROM_LOAD]], %[[ARG_TO]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

void acc_compute() {
  // CHECK: cir.func{{.*}} @acc_compute() {

  int someInt;
  // CHECK-NEXT: %[[SOMEINT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["someInt"]
  float someFloat;
  // CHECK-NEXT: %[[SOMEFLOAT:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["someFloat"]
  struct NoCopyConstruct noCopy;
  // CHECK-NEXT: %[[NOCOPY:.*]] = cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["noCopy"]
  int someIntArr[5];
  // CHECK-NEXT: %[[INTARR:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["someIntArr"]
  float someFloatArr[5];
  // CHECK-NEXT: %[[FLOATARR:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["someFloatArr"]
  struct NoCopyConstruct noCopyArr[5];
  // CHECK-NEXT: %[[NOCOPYARR:.*]] = cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["noCopyArr"]

#pragma acc parallel firstprivate(someInt)
  ;
  // CHECK: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someFloat)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSf -> %[[PRIVATE]] : !cir.ptr<!cir.float>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(noCopy)
  ;
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!rec_NoCopyConstruct>
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(someInt, someFloat, noCopy)
  ;
  // CHECK: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE1]] : !cir.ptr<!s32i>,
  // CHECK-SAME: @firstprivatization__ZTSf -> %[[PRIVATE2]] : !cir.ptr<!cir.float>,
  // CHECK-SAME: @firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!rec_NoCopyConstruct>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial firstprivate(someIntArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(someFloatArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(noCopyArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someIntArr[1], someFloatArr[1], noCopyArr[1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel firstprivate(someIntArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial firstprivate(someFloatArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1:1]"}
  // CHECK-NEXT: acc.serial firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(noCopyArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel firstprivate(someIntArr[1:1], someFloatArr[1:1], noCopyArr[1:1])
  ;
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1:1]"}
  // CHECK-NEXT: acc.parallel firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>)
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
