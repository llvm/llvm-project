// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct CtorDtor {
  int i;
  CtorDtor();
  CtorDtor(const CtorDtor&);
  ~CtorDtor();
};

template<typename T>
void do_things(unsigned A, unsigned B) {
  T OneArr[5];
#pragma acc parallel private(OneArr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSA5_8CtorDtor : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!rec_CtorDtor x 5>, !cir.ptr<!cir.array<!rec_CtorDtor x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
//
// Init Section.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB1_CAST]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR1_LOAD]], %[[UB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OneArr[B])
  ;
#pragma acc parallel private(OneArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_8CtorDtor : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!rec_CtorDtor x 5>, !cir.ptr<!cir.array<!rec_CtorDtor x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<5> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ARR_SIZE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[IDX_LOAD]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[ONE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!rec_CtorDtor x 5>> {{.*}}):
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[DEC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[NEG_ONE]] : (!cir.ptr<!rec_CtorDtor>, !s64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[DEC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;

  T TwoArr[5][5];
#pragma acc parallel private(TwoArr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_8CtorDtor : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!rec_CtorDtor x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// Init Section:
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i 
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB2_CAST]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR2_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
//
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB1_CAST]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR1_LOAD]], %[[UB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i 
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB2_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR2_LOAD]], %[[LB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
//
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT:acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(TwoArr[B][A:B])
  ;
#pragma acc parallel private(TwoArr[A:B][A:B])
  ;
#pragma acc parallel private(TwoArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_8CtorDtor : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!rec_CtorDtor x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 25>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<25> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_CtorDtor x 25>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ARR_SIZE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[IDX_LOAD]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[ONE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> {{.*}}):
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 25>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<24> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_CtorDtor x 25>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[DEC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[NEG_ONE]] : (!cir.ptr<!rec_CtorDtor>, !s64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[DEC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;

  T ThreeArr[5][5][5];
#pragma acc parallel private(ThreeArr[B][B][B])
// CHECK-NEXT:acc.private.recipe @privatization__Bcnt3__ZTSA5_A5_A5_8CtorDtor : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
//
// Init Section:
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB3_CAST]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR3_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB2_CAST]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR2_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_STRIDE_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB1_CAST]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR1_LOAD]], %[[UB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[BOUND1_STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[BOUND1_STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB3_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR3_LOAD]], %[[LB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB2_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR2_LOAD]], %[[LB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_STRIDE_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[BOUND1_STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[BOUND1_STRIDE]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT:acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreeArr[B][B][A:B])
  ;
#pragma acc parallel private(ThreeArr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_A5_8CtorDtor : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// Init Section:
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i 
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB2_CAST]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR2_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
//
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB1_CAST]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(lt, %[[ITR1_LOAD]], %[[UB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<5> : !u64i
// CHECK-NEXT: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[ARR_DECAY]], %[[ARR_SIZE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[ARR_DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[IDX_LOAD]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[ONE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[LAST_ELT]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i 
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB2_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR2_LOAD]], %[[LB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
//
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>>
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[LAST_SUB_ONE:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[LAST_SUB_ONE]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!cir.array<!rec_CtorDtor x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_CtorDtor x 5>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STRIDE]] : !cir.ptr<!cir.array<!rec_CtorDtor x 5>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[ARR_DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[DEC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[NEG_ONE]] : (!cir.ptr<!rec_CtorDtor>, !s64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[DEC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ARR_DECAY]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR2_LOAD]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR2_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT:acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreeArr[B][A:B])
  ;
#pragma acc parallel private(ThreeArr[A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_A5_8CtorDtor : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 125>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<125> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_CtorDtor x 125>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ARR_SIZE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1Ev(%[[IDX_LOAD]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[ONE]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> {{.*}})):
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_CtorDtor x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!rec_CtorDtor x 125>>
// CHECK-NEXT: %[[LAST_IDX:.*]] = cir.const #cir.int<124> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_CtorDtor x 125>> -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[LAST_IDX]] : (!cir.ptr<!rec_CtorDtor>, !u64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[LAST_ELT]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CHECK-NEXT: %[[DEC_STRIDE:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[NEG_ONE]] : (!cir.ptr<!rec_CtorDtor>, !s64i) -> !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.store %[[DEC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[DECAY]]) : !cir.ptr<!rec_CtorDtor>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
}

void use(unsigned A, unsigned B) {
  do_things<CtorDtor>(A, B);
}

