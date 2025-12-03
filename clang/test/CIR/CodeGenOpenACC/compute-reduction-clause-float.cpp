// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

template<typename T>
void acc_compute() {
  T someVar;
  T someVarArr[5];
#pragma acc parallel reduction(+:someVar)
// CHECK: acc.reduction.recipe @reduction_add__ZTSf : !cir.ptr<!cir.float> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[LHSARG]]
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(*:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSf : !cir.ptr<!cir.float> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[LHSARG]]
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }

  ;
#pragma acc parallel reduction(max:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSf : !cir.ptr<!cir.float> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHSARG]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(min:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSf : !cir.ptr<!cir.float> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHSARG]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(&&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSf : !cir.ptr<!cir.float> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[FALSE:.*]] = cir.const #false
// CHECK-NEXT: cir.yield %[[FALSE]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHSARG]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(||:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSf : !cir.ptr<!cir.float> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.float>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.float> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.float> {{.*}})
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[TRUE:.*]] = cir.const #true
// CHECK-NEXT: cir.yield %[[TRUE]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHSARG]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.float>
// CHECK-NEXT: }
  ;

#pragma acc parallel reduction(+:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.zero : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[END_VAL:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[END_VAL]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[LHS_STRIDE]]
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(*:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[CONST_ARRAY:.*]] = cir.const #cir.const_array<[#cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float]> : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[CONST_ARRAY]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[END_VAL:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[END_VAL]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[LHS_STRIDE]]
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(max:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[CONST_ARRAY:.*]] = cir.const #cir.const_array<[#cir.fp<-3.4{{.*}}E+38> : !cir.float, #cir.fp<-3.4{{.*}}E+38> : !cir.float, #cir.fp<-3.4{{.*}}E+38> : !cir.float, #cir.fp<-3.4{{.*}}E+38> : !cir.float, #cir.fp<-3.4{{.*}}E+38> : !cir.float]> : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[CONST_ARRAY]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"]
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[MAX_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[ITR_CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[MAX_IDX]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[ITR_CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(min:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[CONST_ARRAY:.*]] = cir.const #cir.const_array<[#cir.fp<3.4{{.*}}E+38> : !cir.float, #cir.fp<3.4{{.*}}E+38> : !cir.float, #cir.fp<3.4{{.*}}E+38> : !cir.float, #cir.fp<3.4{{.*}}E+38> : !cir.float, #cir.fp<3.4{{.*}}E+38> : !cir.float]> : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[CONST_ARRAY]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"]
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[MAX_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[ITR_CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[MAX_IDX]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[ITR_CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(&&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[CONST_ARRAY:.*]] = cir.const #cir.const_array<[#cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float, #cir.fp<1{{.*}}> : !cir.float]> : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[CONST_ARRAY]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[END_VAL:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[END_VAL]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[FALSE:.*]] = cir.const #false
// CHECK-NEXT: cir.yield %[[FALSE]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(||:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.zero : !cir.array<!cir.float x 5>
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[ALLOCA]] : !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}})
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[END_VAL:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[END_VAL]]) : !s64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !s64i) -> !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[TRUE:.*]] = cir.const #true
// CHECK-NEXT: cir.yield %[[TRUE]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;

#pragma acc parallel reduction(+:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_add__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[LHS_STRIDE]]
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(*:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[LHS_STRIDE]]
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(max:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_max__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LEAST]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"]
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(min:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_min__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[LARGEST]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB]] : index to !u64i
// CHECK-NEXT: %[[UB:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"]
// CHECK-NEXT: cir.store %[[LB_CAST]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float> 
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float, !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[CMP]], true {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RESULT:.*]] = cir.load{{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: cir.yield %[[RESULT]]
// CHECK-NEXT: }) : (!cir.bool) -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[TERNARY]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(&&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_land__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ONE]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[FALSE:.*]] = cir.const #false
// CHECK-NEXT: cir.yield %[[FALSE]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;
#pragma acc parallel reduction(||:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__Bcnt1__ZTSA5_f : !cir.ptr<!cir.array<!cir.float x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!cir.float x 5>, !cir.ptr<!cir.array<!cir.float x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[ZERO]], %[[STRIDE]] : !cir.float, !cir.ptr<!cir.float>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!cir.float x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!cir.float>, !u64i) -> !cir.ptr<!cir.float>
//
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load {{.*}} %[[LHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[LHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: %[[TERNARY:.*]] = cir.ternary(%[[LHS_TO_BOOL]], true {
// CHECK-NEXT: %[[TRUE:.*]] = cir.const #true
// CHECK-NEXT: cir.yield %[[TRUE]] : !cir.bool
// CHECK-NEXT: }, false {
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load {{.*}} %[[RHS_STRIDE]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[RHS_TO_BOOL:.*]] = cir.cast float_to_bool %[[RHS_LOAD]] : !cir.float -> !cir.bool
// CHECK-NEXT: cir.yield %[[RHS_TO_BOOL]] : !cir.bool
// CHECK-NEXT: }) : (!cir.bool) -> !cir.bool
// CHECK-NEXT: %[[RES_TO_VAL:.*]] = cir.cast bool_to_float %[[TERNARY]] : !cir.bool -> !cir.float
// CHECK-NEXT: cir.store{{.*}} %[[RES_TO_VAL]], %[[LHS_STRIDE]] : !cir.float, !cir.ptr<!cir.float>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!cir.float x 5>>
// CHECK-NEXT: }
  ;

#pragma acc parallel reduction(+:someVarArr[1:1])
  ;
#pragma acc parallel reduction(*:someVarArr[1:1])
  ;
#pragma acc parallel reduction(max:someVarArr[1:1])
  ;
#pragma acc parallel reduction(min:someVarArr[1:1])
  ;
#pragma acc parallel reduction(&&:someVarArr[1:1])
  ;
#pragma acc parallel reduction(||:someVarArr[1:1])
  ;
  // CHECK-NEXT: cir.func {{.*}}@_Z11acc_compute
}

void uses() {
  acc_compute<float>();
}
