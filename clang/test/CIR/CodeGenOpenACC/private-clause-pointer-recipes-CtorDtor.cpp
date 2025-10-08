// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct CtorDtor {
  int i;
  CtorDtor();
  CtorDtor(const CtorDtor&);
  ~CtorDtor();
};

template<typename T>
void do_things(unsigned A, unsigned B) {

  T ***ThreePtr;
#pragma acc parallel private(ThreePtr)
// CHECK: acc.private.recipe @privatization__ZTSPPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreePtr[A])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_INT_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_INT_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}, %[[BOUNDS1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][B][B])
  ;
#pragma acc parallel private(ThreePtr[B][B][A:B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}, %[[BOUNDS1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_PTR_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_PTR_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS_2:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_3]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_INT:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS_2]], %[[SIZEOF_INT]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_ALLOCA:.*]] = cir.alloca !rec_CtorDtor, !cir.ptr<!rec_CtorDtor>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>> {{.*}}, %[[BOUNDS1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUNDS3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUNDS3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB3:.*]] = cir.binop(sub, %[[UB3_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB3]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>

// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR3_LOAD]], %[[LB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[PRIVATE]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride(%[[TLA_LOAD]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB2:.*]] = cir.binop(sub, %[[UB2_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB2]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR2_LOAD]], %[[LB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND3_STRIDE_LOAD:.*]] = cir.load %[[BOUND3_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_LOAD]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB1:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB1]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD]] : !cir.ptr<!rec_CtorDtor>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
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
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B][A:B])
  ;

  T **TwoPtr;
#pragma acc parallel private(TwoPtr)
// CHECK: acc.private.recipe @privatization__ZTSPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(TwoPtr[A])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_INT_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_INT_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(TwoPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPP8CtorDtor : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> {{.*}}, %[[BOUNDS1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, ["openacc.private.init"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_INT:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_INT]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !rec_CtorDtor, !cir.ptr<!rec_CtorDtor>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>> {{.*}}, %[[BOUNDS1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUNDS2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUNDS2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB2]] : index to !u64i
// CHECK-NEXT: %[[ITR2:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB2:.*]] = cir.binop(sub, %[[UB2_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB2]], %[[ITR2]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR2_LOAD]], %[[LB2_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
//
// CHECK-NEXT: %[[ITR2_LOAD:.*]] = cir.load %[[ITR2]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[PRIVATE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_CtorDtor>>>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[TLA_STRIDE:.*]] = cir.ptr_stride(%[[TLA_LOAD]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
//
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUNDS1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR1:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB1:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB1]], %[[ITR1]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR1_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR1_LOAD:.*]] = cir.load %[[ITR1]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_STRIDE_LOAD:.*]] = cir.load %[[TLA_STRIDE]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[TLA_STRIDE_LOAD]] : !cir.ptr<!rec_CtorDtor>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
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
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } 
  ;
#pragma acc parallel private(TwoPtr[B][A:B])
  ;
#pragma acc parallel private(TwoPtr[A:B][A:B])
  ;

  T *OnePtr;
#pragma acc parallel private(OnePtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSP8CtorDtor : !cir.ptr<!cir.ptr<!rec_CtorDtor>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!rec_CtorDtor>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(OnePtr[B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSP8CtorDtor : !cir.ptr<!cir.ptr<!rec_CtorDtor>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!rec_CtorDtor>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>>, ["openacc.private.init"]
//
// CHECK-NEXT: %[[INT_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_CTORDTOR:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_CTORDTOR]]) : !u64i
// CHECK-NEXT: %[[INT_VLA_ALLOCA:.*]] = cir.alloca !rec_CtorDtor, !cir.ptr<!rec_CtorDtor>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.ptr<!rec_CtorDtor>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.ptr<!rec_CtorDtor>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i
// CHECK-NEXT: %[[ITR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB:.*]] = cir.binop(sub, %[[UB1_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR_LOAD]], %[[LB1_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[PRIVATE]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[TLA_LOAD]] : !cir.ptr<!rec_CtorDtor>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorD1Ev(%[[STRIDE]]) : (!cir.ptr<!rec_CtorDtor>) -> ()
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[ITR_LOAD]]) : !u64i, !u64i
// CHECK-NEXT: cir.store %[[DEC]], %[[ITR]] : !u64i, !cir.ptr<!u64i>
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
  do_things<CtorDtor>(A, B);
}

