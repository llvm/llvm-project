// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

// Note: unlike the 'private' recipe checks, this is just for spot-checking,
// so this test isn't as comprehensive.  The same code paths are used for
// 'private', so we just count on those to catch the errors.
struct NoOps {
  int i;
  ~NoOps();
};

struct CtorDtor {
  int i;
  CtorDtor();
  ~CtorDtor();
};

void do_things(unsigned A, unsigned B) {
  NoOps ThreeArr[5][5][5];

#pragma acc parallel firstprivate(ThreeArr[B][B][B])
// CHECK:acc.firstprivate.recipe @firstprivatization__Bcnt3__ZTSA5_A5_A5_5NoOps : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>>, ["openacc.firstprivate.init"] {alignment = 4 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
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
// CHECK-NEXT: %[[TLA_DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[ARG_FROM]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE_FROM:.*]] = cir.ptr_stride(%[[TLA_DECAY_FROM]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[TLA_DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[ARG_TO]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE_TO:.*]] = cir.ptr_stride(%[[TLA_DECAY_TO]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE_FROM]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY_FROM]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE_TO]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY_TO]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE_FROM]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND1_STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY_FROM]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE_TO]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND1_STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY_TO]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1ERKS_(%[[BOUND1_STRIDE_TO]], %[[BOUND1_STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_NoOps>, !cir.ptr<!rec_NoOps>) -> ()
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
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
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
// CHECK-NEXT: %[[TLA_DECAY:.*]] = cir.cast array_to_ptrdecay %[[PRIVATE]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>> -> !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride(%[[TLA_DECAY]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>> -> !cir.ptr<!cir.array<!rec_NoOps x 5>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.array<!rec_NoOps x 5>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND2_STRIDE]] : !cir.ptr<!cir.array<!rec_NoOps x 5>> -> !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND1_STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_DECAY]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsD1Ev(%[[BOUND1_STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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

  NoOps ***ThreePtr;
#pragma acc parallel firstprivate(ThreePtr[B][B][A:B])
// CHECK: acc.firstprivate.recipe @firstprivatization__Bcnt3__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TOP_LEVEL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.firstprivate.init"] {alignment = 8 : i64}
// CHECK-NEXT: %[[INT_PTR_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[SIZEOF_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST]], %[[SIZEOF_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_VLA_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
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
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride(%[[INT_PTR_PTR_VLA_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[SRC_IDX]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride(%[[TOP_LEVEL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>
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
// CHECK-NEXT: %[[INT_PTR_PTR_UPPER_BOUND:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UPPER_BOUND_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[INT_PTR_PTR_UPPER_BOUND]] : index to !u64i
// CHECK-NEXT: %[[NUM_ELTS:.*]] = cir.binop(mul, %[[UPPER_BOUND_CAST_2]], %[[UPPER_BOUND_CAST]]) : !u64i
// CHECK-NEXT: %[[SIZEOF_PTR_PTR:.*]] = cir.const #cir.int<8> : !u64i
// CHECK-NEXT: %[[CALC_ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS]], %[[SIZEOF_PTR_PTR]]) : !u64i
// CHECK-NEXT: %[[INT_PTR_PTR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
//
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
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride(%[[INT_PTR_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[SRC_IDX]] : !u64i), !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride(%[[INT_PTR_PTR_VLA_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
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
// CHECK-NEXT: %[[INT_PTR_ALLOCA:.*]] = cir.alloca !rec_NoOps, !cir.ptr<!rec_NoOps>, %[[CALC_ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
//
// Copy array pointer to the original alloca.
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
// CHECK-NEXT: %[[SRC_STRIDE:.*]] = cir.ptr_stride(%[[INT_PTR_ALLOCA]] : !cir.ptr<!rec_NoOps>, %[[SRC_IDX]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[DEST_STRIDE:.*]] = cir.ptr_stride(%[[INT_PTR_PTR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.store %[[SRC_STRIDE]], %[[DEST_STRIDE]] : !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>
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
//
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
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
// CHECK-NEXT: %[[TLA_LOAD_FROM:.*]] = cir.load %[[ARG_FROM]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[BOUND3_STRIDE_FROM:.*]] = cir.ptr_stride(%[[TLA_LOAD_FROM]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[TLA_LOAD_TO:.*]] = cir.load %[[ARG_TO]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[BOUND3_STRIDETO:.*]] = cir.ptr_stride(%[[TLA_LOAD_TO]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_LOAD_FROM:.*]] = cir.load %[[BOUND3_STRIDE_FROM]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_LOAD_FROM]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND3_STRIDE_LOAD_TO:.*]] = cir.load %[[BOUND3_STRIDE_TO]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_LOAD_TO]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_NoOps>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD_FROM:.*]] = cir.load %[[BOUND2_STRIDE_FROM]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD_FROM]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD_TO:.*]] = cir.load %[[BOUND2_STRIDE_TO]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD_TO]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsC1ERKS_(%[[BOUND1_STRIDE_TO]], %[[BOUND1_STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_NoOps>, !cir.ptr<!rec_NoOps>) -> ()
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
//
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
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
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[PRIVATE]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride(%[[TLA_LOAD]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
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
// CHECK-NEXT: %[[BOUND3_STRIDE_LOAD:.*]] = cir.load %[[BOUND3_STRIDE]] : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_LOAD]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_NoOps>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD:.*]] = cir.load %[[BOUND2_STRIDE]] : !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!rec_NoOps>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD]] : !cir.ptr<!rec_NoOps>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_NoOps>
// CHECK-NEXT: cir.call @_ZN5NoOpsD1Ev(%[[STRIDE]]) nothrow : (!cir.ptr<!rec_NoOps>) -> ()
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
  using PtrTArrayTy = CtorDtor*[5];
  PtrTArrayTy *PtrArrayPtr;

#pragma acc parallel firstprivate(PtrArrayPtr[B][B][B])
// CHECK-NEXT: acc.firstprivate.recipe @firstprivatization__Bcnt3__ZTSPA5_P8CtorDtor : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>>, ["openacc.firstprivate.init"] {alignment = 8 : i64}
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i 
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<40> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[UB3_CAST]], %[[ARR_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!cir.ptr<!rec_CtorDtor> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 8 : i64}
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
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride(%[[ARR_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, %[[SRC_IDX]] : !u64i), !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride(%[[TL_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> 
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR_ALLOCA]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>> -> !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[STRIDE:.*]] = cir.ptr_stride(%[[DECAY]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ZERO]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// 
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB1]] : index to !u64i 
// CHECK-NEXT: %[[NUM_ELTS2:.*]] = cir.binop(mul, %[[UB1_CAST]], %[[NUM_ELTS]]) : !u64i
// CHECK-NEXT: %[[ELT_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CHECK-NEXT: %[[ALLOCA_SIZE:.*]] = cir.binop(mul, %[[NUM_ELTS2]], %[[ELT_SIZE]]) : !u64i
// CHECK-NEXT: %[[ARR_ALLOCA2:.*]] = cir.alloca !rec_CtorDtor, !cir.ptr<!rec_CtorDtor>, %[[ALLOCA_SIZE]] : !u64i, ["openacc.init.bounds"] {alignment = 4 : i64}
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
// CHECK-NEXT: %[[SRC:.*]] = cir.ptr_stride(%[[ARR_ALLOCA2]] : !cir.ptr<!rec_CtorDtor>, %[[SRC_IDX]] : !u64i), !cir.ptr<!rec_CtorDtor> 
// CHECK-NEXT: %[[DEST:.*]] = cir.ptr_stride(%[[STRIDE]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>> 
// CHECK-NEXT: cir.store %[[SRC]], %[[DEST]] : !cir.ptr<!rec_CtorDtor>, !cir.ptr<!cir.ptr<!rec_CtorDtor>> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } step {
// CHECK-NEXT: %[[ITR_LOAD:.*]] = cir.load %[[ITR]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[ITR_LOAD]]) : !u64i, !u64i 
// CHECK-NEXT: cir.store %[[INC]], %[[ITR]] : !u64i, !cir.ptr<!u64i> 
// CHECK-NEXT: cir.yield
// CHECK-NEXT: } 
// CHECK-NEXT: } 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: } copy {
// CHECK-NEXT: ^bb0(%[[ARG_FROM:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> {{.*}}, %[[ARG_TO:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
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
// CHECK-NEXT: %[[TLA_LOAD_FROM:.*]] = cir.load %[[ARG_FROM]] : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>>, !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE_FROM:.*]] = cir.ptr_stride(%[[TLA_LOAD_FROM]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
// CHECK-NEXT: %[[TLA_LOAD_TO:.*]] = cir.load %[[ARG_TO]] : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>>, !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE_TO:.*]] = cir.ptr_stride(%[[TLA_LOAD_TO]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY_FROM:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE_FROM]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>> -> !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[BOUND2_STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY_FROM]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY_TO:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE_TO]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>> -> !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[BOUND2_STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY_TO]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
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
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD_FROM:.*]] = cir.load %[[BOUND2_STRIDE_FROM]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE_FROM:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD_FROM]] : !cir.ptr<!rec_CtorDtor>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[BOUND2_STRIDE_LOAD_TO:.*]] = cir.load %[[BOUND2_STRIDE_TO]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: %[[STRIDE_TO:.*]] = cir.ptr_stride(%[[BOUND2_STRIDE_LOAD_TO]] : !cir.ptr<!rec_CtorDtor>, %[[ITR1_LOAD]] : !u64i), !cir.ptr<!rec_CtorDtor>
// CHECK-NEXT: cir.call @_ZN8CtorDtorC1ERKS_(%[[BOUND1_STRIDE_TO]], %[[BOUND1_STRIDE_FROM]]) nothrow : (!cir.ptr<!rec_CtorDtor>, !cir.ptr<!rec_CtorDtor>) -> ()
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
// CHECK-NEXT: } destroy {
// CHECK-NEXT: ^bb0(%[[REF:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> {{.*}}, %[[PRIVATE:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB3:.*]] = acc.get_lowerbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB3]] : index to !u64i
// CHECK-NEXT: %[[UB3:.*]] = acc.get_upperbound %[[BOUND3]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[UB3_CAST:.*]] = builtin.unrealized_conversion_cast %[[UB3]] : index to !u64i
// CHECK-NEXT: %[[ITR3:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["iter"] {alignment = 8 : i64}
// CHECK-NEXT: %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !u64i
// CHECK-NEXT: %[[ONE_BELOW_UB3:.*]] = cir.binop(sub, %[[UB3_CAST]], %[[CONST_ONE]]) : !u64i
// CHECK-NEXT: cir.store %[[ONE_BELOW_UB3]], %[[ITR3]] : !u64i, !cir.ptr<!u64i>
//
// CHECK-NEXT: cir.for : cond {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[COND:.*]] = cir.cmp(ge, %[[ITR3_LOAD]], %[[LB3_CAST]]) : !u64i, !cir.bool
// CHECK-NEXT: cir.condition(%[[COND]])
// CHECK-NEXT: } body {
// CHECK-NEXT: %[[ITR3_LOAD:.*]] = cir.load %[[ITR3]] : !cir.ptr<!u64i>, !u64i
// CHECK-NEXT: %[[TLA_LOAD:.*]] = cir.load %[[PRIVATE]] : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>>, !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
// CHECK-NEXT: %[[BOUND3_STRIDE:.*]] = cir.ptr_stride(%[[TLA_LOAD]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>, %[[ITR3_LOAD]] : !u64i), !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB2:.*]] = acc.get_lowerbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB2_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB2]] : index to !u64i
// CHECK-NEXT: %[[UB2:.*]] = acc.get_upperbound %[[BOUND2]] : (!acc.data_bounds_ty) -> index
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
// CHECK-NEXT: %[[BOUND3_STRIDE_DECAY:.*]] = cir.cast array_to_ptrdecay %[[BOUND3_STRIDE]] : !cir.ptr<!cir.array<!cir.ptr<!rec_CtorDtor> x 5>> -> !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: %[[BOUND2_STRIDE:.*]] = cir.ptr_stride(%[[BOUND3_STRIDE_DECAY]] : !cir.ptr<!cir.ptr<!rec_CtorDtor>>, %[[ITR2_LOAD]] : !u64i), !cir.ptr<!cir.ptr<!rec_CtorDtor>>
// CHECK-NEXT: cir.scope {
// CHECK-NEXT: %[[LB1:.*]] = acc.get_lowerbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
// CHECK-NEXT: %[[LB1_CAST:.*]] = builtin.unrealized_conversion_cast %[[LB1]] : index to !u64i
// CHECK-NEXT: %[[UB1:.*]] = acc.get_upperbound %[[BOUND1]] : (!acc.data_bounds_ty) -> index
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
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: }
  ;
}
