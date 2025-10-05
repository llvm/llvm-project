// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

template<typename T>
void acc_loop() {
  T someVar;
  T someVarArr[5];
#pragma acc loop reduction(+:someVar)
// CHECK: acc.reduction.recipe @reduction_add__ZTSi : !cir.ptr<!s32i> reduction_operator <add> init {
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
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(*:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSi : !cir.ptr<!s32i> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(max:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSi : !cir.ptr<!s32i> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(min:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSi : !cir.ptr<!s32i> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSi : !cir.ptr<!s32i> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(|:someVar)
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
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(^:someVar)
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

  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSi : !cir.ptr<!s32i> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!s32i>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!s32i> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!s32i> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!s32i>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(||:someVar)
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
  for(int i=0;i < 5; ++i);

#pragma acc loop reduction(+:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
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
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(*:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[DECAY]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ONE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[TWO_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[THREE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(max:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[DECAY]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ONE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[TWO_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[THREE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(min:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[DECAY]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ONE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[TWO_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[THREE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[DECAY]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ONE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[TWO_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[THREE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(|:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
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
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(^:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
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
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[DECAY]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ONE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[TWO_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[THREE_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[FOUR_IDX]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[NEXT_ELT]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(||:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
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
  for(int i=0;i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_add__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_max__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_min__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ALL_ONES]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_land__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.reduction.init"]
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!s32i>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!s32i x 5>>
// CHECK-NEXT: }
  for(int i=0;i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(|:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(^:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[1:1])
  for(int i=0;i < 5; ++i);
  // CHECK-NEXT: cir.func {{.*}}@_Z8acc_loop
}

void uses() {
  acc_loop<int>();
}
