// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s


// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[LAST_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[LAST_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[LAST_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>>), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[LAST_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[TEMP_LOAD]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride(%[[TEMP_LOAD]] : !cir.ptr<!s32i>, %[[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }


// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSi : !cir.ptr<!s32i> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSi : !cir.ptr<!s32i> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSi : !cir.ptr<!s32i> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSi : !cir.ptr<!s32i> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSi : !cir.ptr<!s32i> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSi : !cir.ptr<!s32i> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSi : !cir.ptr<!s32i> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSi : !cir.ptr<!s32i> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init"]
// TODO OpenACC: Expecting an initialization to... SOME value here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSi : !cir.ptr<!s32i> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }

template<typename T>
void acc_loop() {
  T someVar;
  T someVarArr[5];
#pragma acc loop reduction(+:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVar)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVar)
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr)
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr)
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[2])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[2])
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
}

void uses() {
  acc_loop<int>();
}
