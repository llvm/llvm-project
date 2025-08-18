// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoCopyConstruct {};

struct CopyConstruct {
  CopyConstruct() = default;
  CopyConstruct(const CopyConstruct&);
};

struct NonDefaultCtor {
  NonDefaultCtor();
};

struct HasDtor {
  ~HasDtor();
};

// CHECK: acc.firstprivate.recipe @firstprivatization__ZTSA5_7HasDtor : !cir.ptr<!cir.array<!rec_HasDtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_HasDtor x 5>, !cir.ptr<!cir.array<!rec_HasDtor x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_HasDtor x 5>> {{.*}}):
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ARRPTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARG]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[ELEM:.*]] = cir.ptr_stride(%[[ARRPTR]] : !cir.ptr<!rec_HasDtor>, %[[LAST_IDX]] : !u64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>, ["__array_idx"]
// CHECK-NEXT: cir.store %[[ELEM]], %[[ITR]] : !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[ELEM_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.call @_ZN7HasDtorD1Ev(%[[ELEM_LOAD]]) nothrow : (!cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] =  cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[PREVELEM:.*]] = cir.ptr_stride(%[[ELEM_LOAD]] : !cir.ptr<!rec_HasDtor>, %[[NEG_ONE]] : !s64i), !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: cir.store %[[PREVELEM]], %[[ITR]] : !cir.ptr<!rec_HasDtor>, !cir.ptr<!cir.ptr<!rec_HasDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[ELEM_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!cir.ptr<!rec_HasDtor>>, !cir.ptr<!rec_HasDtor>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[ELEM_LOAD]], %[[ARRPTR]]) : !cir.ptr<!rec_HasDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_14NonDefaultCtor : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}):
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_NonDefaultCtor x 5>, !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FORM:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_13CopyConstruct : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_CopyConstruct x 5>, !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FORM:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_15NoCopyConstruct : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FORM:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FORM:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FORM:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS7HasDtor : !cir.ptr<!rec_HasDtor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_HasDtor, !cir.ptr<!rec_HasDtor>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_HasDtor> {{.*}}):
// CHECK-NEXT: cir.call @_ZN7HasDtorD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_HasDtor>) -> ()
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS14NonDefaultCtor : !cir.ptr<!rec_NonDefaultCtor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}):
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_NonDefaultCtor, !cir.ptr<!rec_NonDefaultCtor>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_NonDefaultCtor> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS13CopyConstruct : !cir.ptr<!rec_CopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_CopyConstruct, !cir.ptr<!rec_CopyConstruct>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_CopyConstruct> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTS15NoCopyConstruct : !cir.ptr<!rec_NoCopyConstruct> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
// CHECK-NEXT: cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!rec_NoCopyConstruct> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSf : !cir.ptr<!cir.float> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.float> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__ZTSi : !cir.ptr<!s32i> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i> {{.*}}):
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!s32i> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!s32i> {{.*}}):
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

extern "C" void acc_combined() {
  // CHECK: cir.func{{.*}} @acc_combined() {

  int someInt;
  // CHECK-NEXT: %[[SOMEINT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["someInt"]
  float someFloat;
  // CHECK-NEXT: %[[SOMEFLOAT:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["someFloat"]
  NoCopyConstruct noCopy;
  // CHECK-NEXT: %[[NOCOPY:.*]] = cir.alloca !rec_NoCopyConstruct, !cir.ptr<!rec_NoCopyConstruct>, ["noCopy"]
  CopyConstruct hasCopy;
  // CHECK-NEXT: %[[HASCOPY:.*]] = cir.alloca !rec_CopyConstruct, !cir.ptr<!rec_CopyConstruct>, ["hasCopy"]
  NonDefaultCtor notDefCtor;
  // CHECK-NEXT: %[[NOTDEFCTOR:.*]] = cir.alloca !rec_NonDefaultCtor, !cir.ptr<!rec_NonDefaultCtor>, ["notDefCtor", init]
  HasDtor dtor;
  // CHECK-NEXT: %[[DTOR:.*]] = cir.alloca !rec_HasDtor, !cir.ptr<!rec_HasDtor>, ["dtor"]
  int someIntArr[5];
  // CHECK-NEXT: %[[INTARR:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["someIntArr"]
  float someFloatArr[5];
  // CHECK-NEXT: %[[FLOATARR:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["someFloatArr"]
  NoCopyConstruct noCopyArr[5];
  // CHECK-NEXT: %[[NOCOPYARR:.*]] = cir.alloca !cir.array<!rec_NoCopyConstruct x 5>, !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>, ["noCopyArr"]
  CopyConstruct hasCopyArr[5];
  // CHECK-NEXT: %[[HASCOPYARR:.*]] = cir.alloca !cir.array<!rec_CopyConstruct x 5>, !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>, ["hasCopyArr"]
  NonDefaultCtor notDefCtorArr[5];
  // CHECK-NEXT: %[[NOTDEFCTORARR:.*]] = cir.alloca !cir.array<!rec_NonDefaultCtor x 5>, !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>, ["notDefCtorArr", init]
  HasDtor dtorArr[5];
  // CHECK-NEXT: %[[DTORARR:.*]] = cir.alloca !cir.array<!rec_HasDtor x 5>, !cir.ptr<!cir.array<!rec_HasDtor x 5>>, ["dtorArr"]
  // CHECK-NEXT: cir.call @_ZN14NonDefaultCtorC1Ev(%[[NOTDEFCTOR]]) : (!cir.ptr<!rec_NonDefaultCtor>) -> ()

#pragma acc parallel loop firstprivate(someInt)
  for(int i = 0; i < 5; ++i);
  // CHECK: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(someFloat)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSf -> %[[PRIVATE]] : !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop firstprivate(noCopy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!rec_NoCopyConstruct>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(hasCopy)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPY]] : !cir.ptr<!rec_CopyConstruct>) -> !cir.ptr<!rec_CopyConstruct> {name = "hasCopy"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTS13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!rec_CopyConstruct>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(notDefCtor)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTOR]] : !cir.ptr<!rec_NonDefaultCtor>) -> !cir.ptr<!rec_NonDefaultCtor> {name = "notDefCtor"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTS14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!rec_NonDefaultCtor>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(dtor)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTOR]] : !cir.ptr<!rec_HasDtor>) -> !cir.ptr<!rec_HasDtor> {name = "dtor"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTS7HasDtor -> %[[PRIVATE]] : !cir.ptr<!rec_HasDtor>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop firstprivate(someInt, someFloat, noCopy, hasCopy, notDefCtor, dtor)
  for(int i = 0; i < 5; ++i);
  // CHECK: %[[PRIVATE1:.*]] = acc.firstprivate varPtr(%[[SOMEINT]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "someInt"}
  // CHECK-NEXT: %[[PRIVATE2:.*]] = acc.firstprivate varPtr(%[[SOMEFLOAT]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "someFloat"}
  // CHECK-NEXT: %[[PRIVATE3:.*]] = acc.firstprivate varPtr(%[[NOCOPY]] : !cir.ptr<!rec_NoCopyConstruct>) -> !cir.ptr<!rec_NoCopyConstruct> {name = "noCopy"}
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPY]] : !cir.ptr<!rec_CopyConstruct>) -> !cir.ptr<!rec_CopyConstruct> {name = "hasCopy"}
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTOR]] : !cir.ptr<!rec_NonDefaultCtor>) -> !cir.ptr<!rec_NonDefaultCtor> {name = "notDefCtor"}
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTOR]] : !cir.ptr<!rec_HasDtor>) -> !cir.ptr<!rec_HasDtor> {name = "dtor"}
  // CHECK: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSi -> %[[PRIVATE1]] : !cir.ptr<!s32i>,
  // CHECK-SAME: @firstprivatization__ZTSf -> %[[PRIVATE2]] : !cir.ptr<!cir.float>,
  // CHECK-SAME: @firstprivatization__ZTS15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!rec_NoCopyConstruct>,
  // CHECK-SAME: @firstprivatization__ZTS13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!rec_CopyConstruct>,
  // CHECK-SAME: @firstprivatization__ZTS14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!rec_NonDefaultCtor>,
  // CHECK-SAME: @firstprivatization__ZTS7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!rec_HasDtor>)
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop firstprivate(someIntArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1]"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(someFloatArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(noCopyArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1]"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(hasCopyArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(notDefCtorArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(dtorArr[1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(someIntArr[1], someFloatArr[1], noCopyArr[1], hasCopyArr[1], notDefCtorArr[1], dtorArr[1])
  for(int i = 0; i < 5; ++i);
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
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CONST]] : i64) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1]"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop firstprivate(someIntArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[INTARR]] : !cir.ptr<!cir.array<!s32i x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!s32i x 5>> {name = "someIntArr[1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE]] : !cir.ptr<!cir.array<!s32i x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(someFloatArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[FLOATARR]] : !cir.ptr<!cir.array<!cir.float x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!cir.float x 5>> {name = "someFloatArr[1:1]"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSA5_f -> %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.float x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(noCopyArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOCOPYARR]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>> {name = "noCopyArr[1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop firstprivate(hasCopyArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1:1]"}
  // CHECK-NEXT: acc.serial combined(loop) firstprivate(@firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) {
  // CHECK-NEXT: acc.loop combined(serial)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(notDefCtorArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(dtorArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) {
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop firstprivate(someIntArr[1:1], someFloatArr[1:1], noCopyArr[1:1], hasCopyArr[1:1], notDefCtorArr[1:1], dtorArr[1:1])
  for(int i = 0; i < 5; ++i);
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
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE4:.*]] = acc.firstprivate varPtr(%[[HASCOPYARR]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_CopyConstruct x 5>> {name = "hasCopyArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE5:.*]] = acc.firstprivate varPtr(%[[NOTDEFCTORARR]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>> {name = "notDefCtorArr[1:1]"}
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE]] : !s32i to si32
  // CHECK-NEXT: %[[ZERO_CONST:.*]] = arith.constant 0
  // CHECK-NEXT: %[[ONE_CONST2:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ONE_CAST]] : si32) extent(%[[ONE_CAST2]] : si32) stride(%[[ONE_CONST2]] : i64) startIdx(%[[ZERO_CONST]] : i64)
  // CHECK-NEXT: %[[PRIVATE6:.*]] = acc.firstprivate varPtr(%[[DTORARR]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.array<!rec_HasDtor x 5>> {name = "dtorArr[1:1]"}
  // CHECK-NEXT: acc.parallel combined(loop) firstprivate(@firstprivatization__ZTSA5_i -> %[[PRIVATE1]] : !cir.ptr<!cir.array<!s32i x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_f -> %[[PRIVATE2]] : !cir.ptr<!cir.array<!cir.float x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_15NoCopyConstruct -> %[[PRIVATE3]] : !cir.ptr<!cir.array<!rec_NoCopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_13CopyConstruct -> %[[PRIVATE4]] : !cir.ptr<!cir.array<!rec_CopyConstruct x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_14NonDefaultCtor -> %[[PRIVATE5]] : !cir.ptr<!cir.array<!rec_NonDefaultCtor x 5>>,
  // CHECK-SAME: @firstprivatization__ZTSA5_7HasDtor -> %[[PRIVATE6]] : !cir.ptr<!cir.array<!rec_HasDtor x 5>>)
  // CHECK-NEXT: acc.loop combined(parallel)
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
