// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoOps { int i = 0; };

template<typename T>
void do_things(unsigned A, unsigned B) {
  T OneArr[5];
#pragma acc parallel private(OneArr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSA5_5NoOps : !cir.ptr<!cir.array<!rec_NoOps x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!rec_NoOps x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
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
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[TLA_DECAY]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OneArr[B])
  ;
#pragma acc parallel private(OneArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_5NoOps : !cir.ptr<!cir.array<!rec_NoOps x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!rec_NoOps x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<5> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_NoOps>, %[[ARR_SIZE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride(%[[IDX_LOAD]] : !cir.ptr<!rec_NoOps>, %[[ONE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_NoOps>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;

  T TwoArr[5][5];
#pragma acc parallel private(TwoArr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!rec_NoOps x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
//
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
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[TLA_DECAY]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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
// CHECK-NEXT:}
  ;
#pragma acc parallel private(TwoArr[B][A:B])
  ;
#pragma acc parallel private(TwoArr[A:B][A:B])
  ;
#pragma acc parallel private(TwoArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!rec_NoOps x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 25>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<25> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_NoOps x 25>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_NoOps>, %[[ARR_SIZE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride(%[[IDX_LOAD]] : !cir.ptr<!rec_NoOps>, %[[ONE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_NoOps>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;

  T ThreeArr[5][5][5];
#pragma acc parallel private(ThreeArr[B][B][B])
// CHECK-NEXT:acc.private.recipe @privatization__Bcnt3__ZTSA5_A5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
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
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride(%[[TLA_DECAY]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND1_STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[BOUND1_STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreeArr[B][B][A:B])
  ;
#pragma acc parallel private(ThreeArr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
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
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[TLA_DECAY]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<5> : !u64i
// CHECK-NEXT: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[LAST_ELT:.*]] = cir.ptr_stride(%[[ARR_DECAY]] : !cir.ptr<!rec_NoOps>, %[[ARR_SIZE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[ARR_DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride(%[[IDX_LOAD]] : !cir.ptr<!rec_NoOps>, %[[ONE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[LAST_ELT]]) : !cir.ptr<!rec_NoOps>, !cir.bool
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
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreeArr[B][A:B])
  ;
#pragma acc parallel private(ThreeArr[A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>>, ["openacc.private.init", init] {alignment = 16 : i64}
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast bitcast %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 125>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<125> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BITCAST]] : !cir.ptr<!cir.array<!rec_NoOps x 125>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ONE_PAST_LAST_ELT:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!rec_NoOps>, %[[ARR_SIZE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ARR_IDX:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC_STRIDE:.*]] = cir.ptr_stride(%[[IDX_LOAD]] : !cir.ptr<!rec_NoOps>, %[[ONE]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.store %[[INC_STRIDE]], %[[ARR_IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[ARR_IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ONE_PAST_LAST_ELT]]) : !cir.ptr<!rec_NoOps>, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
}

void use(unsigned A, unsigned B) {
  do_things<NoOps>(A, B);
}

