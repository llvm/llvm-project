// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct DefaultOperators {
  int i;
  unsigned u;
  float f;
  double d;
  bool b;
};

struct DefaultOperatorsNoFloats {
  int i;
  unsigned int u;
  bool b;
};

template<typename T>
void acc_loop() {
  T someVar;
  T someVarArr[5];
  struct DefaultOperatorsNoFloats someVarNoFloats;
  struct DefaultOperatorsNoFloats someVarArrNoFloats[5];
#pragma acc loop reduction(+:someVar)
// CHECK: acc.reduction.recipe @reduction_add__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[ADD]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[MUL]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTS24DefaultOperatorsNoFloats : !cir.ptr<!rec_DefaultOperatorsNoFloats> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperatorsNoFloats, !cir.ptr<!rec_DefaultOperatorsNoFloats>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}})
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[AND]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTS24DefaultOperatorsNoFloats : !cir.ptr<!rec_DefaultOperatorsNoFloats> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperatorsNoFloats, !cir.ptr<!rec_DefaultOperatorsNoFloats>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}})
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[OR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTS24DefaultOperatorsNoFloats : !cir.ptr<!rec_DefaultOperatorsNoFloats> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperatorsNoFloats, !cir.ptr<!rec_DefaultOperatorsNoFloats>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperatorsNoFloats> {{.*}}) 
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHSARG]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[XOR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVar)
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__ZTS16DefaultOperators : !cir.ptr<!rec_DefaultOperators> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_DefaultOperators>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !rec_DefaultOperators, !cir.ptr<!rec_DefaultOperators>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[ALLOCA]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[ALLOCA]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[ALLOCA]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[ALLOCA]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[ALLOCA]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!rec_DefaultOperators> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_add__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_DefaultOperators>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
//
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[ADD]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ONE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
//
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[MUL]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr)

// CHECK-NEXT: acc.reduction.recipe @reduction_max__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[LEAST_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LEAST_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_min__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[LARGEST_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LARGEST_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArrNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ALL_ONES_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ALL_ONES_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}})
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
//
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[AND]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArrNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}})
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
//
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[OR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArrNoFloats)
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperatorsNoFloats>>, !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_DefaultOperatorsNoFloats>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}})
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !s64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
//
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[XOR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!s64i>, !s64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !s64i, !s64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !s64i, !cir.ptr<!s64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr)
// CHECK-NEXT: acc.reduction.recipe @reduction_land__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[DECAY]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[DECAY]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[DECAY]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[DECAY]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[DECAY]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE_IDX:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ONE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[TWO_IDX:.*]] = cir.const #cir.int<2> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[TWO_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
//
// CHECK-NEXT: %[[THREE_IDX:.*]] = cir.const #cir.int<3> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[THREE_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[FOUR_IDX:.*]] = cir.const #cir.int<4> : !s64i
// CHECK-NEXT: %[[NEXT_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[FOUR_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[NEXT_ELT]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[NEXT_ELT]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[NEXT_ELT]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[NEXT_ELT]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[NEXT_ELT]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr)
// CHECK: acc.reduction.recipe @reduction_lor__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}})
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init", init]
// CHECK-NEXT: %[[TEMP_ITR:.*]] = cir.alloca !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, ["arrayinit.temp"]
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.store {{.*}} %[[DECAY]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<5> : !s64i
// CHECK-NEXT: %[[END_ITR:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[TEMP_LOAD]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[TEMP_LOAD]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[TEMP_LOAD]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[TEMP_LOAD]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[TEMP_LOAD]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
//
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CHECK-NEXT: %[[NEXT_ITEM:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ONE]] : (!cir.ptr<!rec_DefaultOperators>, !s64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: cir.store {{.*}} %[[NEXT_ITEM]], %[[TEMP_ITR]] : !cir.ptr<!rec_DefaultOperators>, !cir.ptr<!cir.ptr<!rec_DefaultOperators>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load {{.*}} %[[TEMP_ITR]] : !cir.ptr<!cir.ptr<!rec_DefaultOperators>>, !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[TEMP_LOAD]], %[[END_ITR]]) : !cir.ptr<!rec_DefaultOperators>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
//
// CHECK-NEXT: } combiner {
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}})
// TODO OpenACC: Expecting combination operation here
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_add__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <add> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ADD]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[ADD]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_mul__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <mul> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) nsw : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.float>, !cir.float
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[MUL]], %[[GET_MEM_LHS]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) nsw : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[MUL]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_max__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <max> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<-2147483648> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #cir.fp<-1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LEAST:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[LEAST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_min__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <min> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<2147483647> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<3.4{{.*}}E+38> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #cir.fp<1.7{{.*}}E+308> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[LARGEST:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[LARGEST]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArrNoFloats[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_iand__Bcnt1__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <iand> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<-1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ALL_ONES:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ALL_ONES]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[AND]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[AND:.*]] = cir.binop(and, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[AND]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArrNoFloats[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_ior__Bcnt1__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <ior> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[OR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[OR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArrNoFloats[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_xor__Bcnt1__ZTSA5_24DefaultOperatorsNoFloats : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> reduction_operator <xor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperatorsNoFloats x 5>, !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
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
// CHECK-NEXT: %[[LHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[LHS_STRIDE:.*]] = cir.ptr_stride %[[LHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_DECAY:.*]] = cir.cast array_to_ptrdecay %[[RHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>> -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[RHS_STRIDE:.*]] = cir.ptr_stride %[[RHS_DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperatorsNoFloats>, !u64i) -> !cir.ptr<!rec_DefaultOperatorsNoFloats>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!u32i>, !u32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_LOAD]], %[[RHS_LOAD]]) : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[XOR]], %[[GET_MEM_LHS]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_MEM_LHS:.*]] = cir.get_member %[[LHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[GET_MEM_RHS:.*]] = cir.get_member %[[RHS_STRIDE]][2] {name = "b"} : !cir.ptr<!rec_DefaultOperatorsNoFloats> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[RHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_RHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[RHS_INT_CAST:.*]] = cir.cast bool_to_int %[[RHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[LHS_LOAD:.*]] = cir.load{{.*}} %[[GET_MEM_LHS]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK-NEXT: %[[LHS_INT_CAST:.*]] = cir.cast bool_to_int %[[LHS_LOAD]] : !cir.bool -> !s32i
// CHECK-NEXT: %[[XOR:.*]] = cir.binop(xor, %[[LHS_INT_CAST]], %[[RHS_INT_CAST]]) : !s32i
// CHECK-NEXT: %[[RES_TO_BOOL_CAST:.*]] = cir.cast int_to_bool %[[XOR]] : !s32i -> !cir.bool
// CHECK-NEXT: cir.store {{.*}} %[[RES_TO_BOOL_CAST]], %[[GET_MEM_LHS]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperatorsNoFloats x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_land__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <land> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.fp<1{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ONE:.*]] = cir.const #true
// CHECK-NEXT: cir.store {{.*}} %[[ONE]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[2])
// CHECK-NEXT: acc.reduction.recipe @reduction_lor__Bcnt1__ZTSA5_16DefaultOperators : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> reduction_operator <lor> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>{{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: %[[ALLOCA:.*]] = cir.alloca !cir.array<!rec_DefaultOperators x 5>, !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>, ["openacc.reduction.init"]
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ALLOCA]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR_LOAD]] : (!cir.ptr<!rec_DefaultOperators>, !u64i) -> !cir.ptr<!rec_DefaultOperators>
// CHECK-NEXT: %[[GET_I:.*]] = cir.get_member %[[STRIDE]][0] {name = "i"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_I]] : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT: %[[GET_U:.*]] = cir.get_member %[[STRIDE]][1] {name = "u"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!u32i>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_U]] : !u32i, !cir.ptr<!u32i>
// CHECK-NEXT: %[[GET_F:.*]] = cir.get_member %[[STRIDE]][2] {name = "f"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.float>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.float
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_F]] : !cir.float, !cir.ptr<!cir.float>
// CHECK-NEXT: %[[GET_D:.*]] = cir.get_member %[[STRIDE]][3] {name = "d"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.double>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.fp<0{{.*}}> : !cir.double
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_D]] : !cir.double, !cir.ptr<!cir.double>
// CHECK-NEXT: %[[GET_B:.*]] = cir.get_member %[[STRIDE]][4] {name = "b"} : !cir.ptr<!rec_DefaultOperators> -> !cir.ptr<!cir.bool>
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #false
// CHECK-NEXT: cir.store {{.*}} %[[ZERO]], %[[GET_B]] : !cir.bool, !cir.ptr<!cir.bool>
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
// CHECK-NEXT: ^bb0(%[[LHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[RHSARG:.*]]: !cir.ptr<!cir.array<!rec_DefaultOperators x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty{{.*}}))
// CHECK-NEXT: acc.yield %[[LHSARG]] : !cir.ptr<!cir.array<!rec_DefaultOperators x 5>>
// CHECK-NEXT: }
  for(int i = 0; i < 5; ++i);

#pragma acc loop reduction(+:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(*:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(max:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(min:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&:someVarArrNoFloats[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(|:someVarArrNoFloats[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(^:someVarArrNoFloats[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(&&:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
#pragma acc loop reduction(||:someVarArr[1:1])
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: cir.func {{.*}}@_Z8acc_loop
}

void uses() {
  acc_loop<DefaultOperators>();
}
