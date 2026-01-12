// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

template<typename T>
void do_things(unsigned A, unsigned B) {

  T ***ThreePtr;
#pragma acc parallel private(ThreePtr)
// CHECK: acc.private.recipe @privatization__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreePtr[A])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_INT_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_INT_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.yield
//
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
#pragma acc parallel private(ThreePtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}

// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_BOUND_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_PTR_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_PTR_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_PTR_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_BOUND_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_PTR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_PTR_VLA_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS_2:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_3]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_INT:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS_2]], %[[SIZEOF_INT]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[NUM_ELTS]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_3]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_PTR_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B][A:B])
  ;

  T **TwoPtr;
#pragma acc parallel private(TwoPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPi : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(TwoPtr[A])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPPi : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_INT_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_INT_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(TwoPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPi : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !u64i) -> !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_INT:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_INT]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_BOUND_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[INT_PTR_VLA_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
//
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.yield
// CHECK-NEXT: }
// CHECK-NEXT: }
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(TwoPtr[B][A:B])
  ;
#pragma acc parallel private(TwoPtr[A:B][A:B])
  ;

  T *OnePtr;
#pragma acc parallel private(OnePtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPi : !cir.ptr<!cir.ptr<!s32i>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!s32i>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(OnePtr[B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSPi : !cir.ptr<!cir.ptr<!s32i>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!s32i>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_INT:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_INT]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// Copy array pointer to the original alloca.
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["itr"] {alignment = 8 : i64}
// CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK-NEXT: cir.store %[[ZERO]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[UPPER_LIMIT:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[CMP:.*]] = cir.cmp(lt, %[[ITR_LOAD]], %[[UPPER_LIMIT]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[CMP]])
//
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[SRC_IDX:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[ITR_LOAD]]) : !u64i
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride %[[INT_VLA_ALLOCA]], %[[SRC_IDX]] : (!cir.ptr<!s32i>, !u64i) -> !cir.ptr<!s32i>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride %[[TOP_LEVEL_ALLOCA]], %[[ITR_LOAD]] : (!cir.ptr<!cir.ptr<!s32i>>, !u64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK-NEXT: cir.yield
//
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
#pragma acc parallel private(OnePtr[A:B])
  ;
}

void use(unsigned A, unsigned B) {
  do_things<int>(A, B);
}

