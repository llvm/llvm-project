// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoOps { int i = 0; };

template<typename T>
void do_things(unsigned A, unsigned B) {
  T *OnePtr;
#pragma acc parallel private(OnePtr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["openacc.private.init"] {alignment = 8 : i64} 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// Init Section:
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
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ELT_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[ELT_STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR1_LOAD]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR1_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OnePtr[B])
  ;
#pragma acc parallel private(OnePtr)
// CHECK: acc.private.recipe @privatization__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
  ;

  T **TwoPtr;
#pragma acc parallel private(TwoPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[UB2_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
// Init Section.
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
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
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
// CHECK-NEXT: }
  ;
#pragma acc parallel private(TwoPtr[B][A:B])
  ;
#pragma acc parallel private(TwoPtr[A:B][A:B])
  ;
#pragma acc parallel private(TwoPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  T ***ThreePtr;
#pragma acc parallel private(ThreePtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[UB3_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS2:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS2]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA3:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[NUM_ELTS]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA3]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// Init:
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB3_CAST]], %[[ITR3]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR3_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] :  !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
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
// CHECK-NEXT: %[[BOUND3_LOAD:.*]] = cir.load %[[BOUND3_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
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
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[UB2_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;


  T *ArrayOfPtr[5];
#pragma acc parallel private(ArrayOfPtr[B][A:B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_P5NoOps : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!rec_NoOps> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, ["openacc.private.init"] {alignment = 8 : i64} 
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
//
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[TL_DEREF:.*]] = cir.ptr_stride %[[DECAY]], %[[ZERO]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[UB2_CAST]]) : !u64i
// CHECK-NEXT: %[[ELT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ELT_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64} 
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_DEREF]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
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
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ELT_STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[ELT_STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ArrayOfPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtr[B][B])
  ;
#pragma acc parallel private(ArrayOfPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_P5NoOps : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!rec_NoOps> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, ["openacc.private.init"] {alignment = 16 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
  ;

  using TArrayTy = T[5];
  TArrayTy *PtrToArrays;
#pragma acc parallel private(PtrToArrays[B][B])
  ;
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, ["openacc.private.init"] {alignment = 8 : i64} 
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<20> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64} 
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
// Init Section
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
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ELT_STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[ELT_STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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
// CHECK-NEXT: }
#pragma acc parallel private(PtrToArrays[B][A:B])
  ;
#pragma acc parallel private(PtrToArrays[A:B][A:B])
  ;
#pragma acc parallel private(PtrToArrays)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  T **ArrayOfPtrPtr[5];
#pragma acc parallel private(ArrayOfPtrPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i 
//
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[TL_DEREF:.*]] = cir.ptr_stride %[[DECAY]], %[[ZERO]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[UB3_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64} 
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_DEREF]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS2:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[ELT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS2]], %[[ELT_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[NUM_ELTS]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: }
// Init Section.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: cir.store %[[LB3_CAST]], %[[ITR3]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR3_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_DECAY]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_LOAD:.*]] = cir.load %[[BOUND3_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_STRIDE_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
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
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
//
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[TL_DEREF:.*]] = cir.ptr_stride %[[DECAY]], %[[ZERO]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[UB2_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64} 
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_DEREF]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  TArrayTy **PtrPtrToArray;
#pragma acc parallel private(PtrPtrToArray[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[UB3_CAST]]) : !u64i
// CHECK-NEXT: %[[ELT_SIZE:.*]] = cir.const #cir.int<20> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ELT_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: }
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
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR3_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
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
// CHECK-NEXT: %[[BOUND3_LOAD:.*]] = cir.load %[[BOUND3_STRIDE]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_DECAY]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
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
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrPtrToArray[B][B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[UB2_CAST]]) : !u64i
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<20> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: }
// Init Section.
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
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!cir.array<!rec_NoOps x 5>>, !u64i) -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<5> : !u64i 
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[ELT:.*]] = cir.ptr_stride %[[DECAY]], %[[ARR_SIZE]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[IDX:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["__array_idx"] {alignment = 1 : i64}
// CHECK-NEXT: cir.store %[[DECAY]], %[[IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.do {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1Ev(%[[IDX_LOAD]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
// CHECK-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.ptr_stride %[[IDX_LOAD]], %[[ONE]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.store %[[INC]], %[[IDX]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } while {
// CHECK-NEXT: %[[IDX_LOAD:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[IDX_LOAD]], %[[ELT]]) : !cir.ptr<!rec_NoOps>, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
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
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  using PtrTArrayTy = T*[5];
  PtrTArrayTy *PtrArrayPtr;

#pragma acc parallel private(PtrArrayPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<40> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!rec_NoOps> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[UB3_CAST]]) : !u64i
//
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[DECAY]], %[[ZERO]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
//
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS2:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[ELT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS2]], %[[ELT_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[NUM_ELTS]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA2]], %[[SRC_IDX]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[STRIDE]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
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
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR3_LOAD]], %[[UB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride %[[TLA_LOAD]], %[[ITR3_LOAD]] : (!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> -> !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride %[[BOUND3_STRIDE_DECAY]], %[[ITR2_LOAD]] : (!cir.ptr<!cir.ptr<!rec_NoOps>>, !u64i) -> !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride %[[BOUND2_STRIDE_LOAD]], %[[ITR1_LOAD]] : (!cir.ptr<!rec_NoOps>, !u64i) -> !cir.ptr<!rec_NoOps>
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
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR3_LOAD]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR3_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrArrayPtr[B][B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<40> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!rec_NoOps> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPP_BOUND:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPP_BOUND]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]]) 
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UB2_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride %[[ARR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !u64i) -> !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride %[[TL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> 
// CHECK-NEXT: cir.yield 
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrArrayPtr[B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
}

void use(unsigned A, unsigned B) {
  do_things<NoOps>(A, B);
}

