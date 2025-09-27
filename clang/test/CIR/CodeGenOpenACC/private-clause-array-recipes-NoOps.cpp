// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoOps { int i = 0; };

template<typename T>
void do_things(unsigned A, unsigned B) {
  T OneArr[5];
#pragma acc parallel private(OneArr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSA5_5NoOps : !cir.ptr<!cir.array<!rec_NoOps x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!rec_NoOps x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!rec_NoOps x 5>, !cir.ptr<!cir.array<!rec_NoOps x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// TODO: Add Init here.
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
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!rec_NoOps x 5>>), !cir.ptr<!rec_NoOps>
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
// TODO: Add Init here.
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
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast(bitcast, %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_NoOps x 5> x 5>>), !cir.ptr<!cir.array<!rec_NoOps x 25>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<25> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[BITCAST]] : !cir.ptr<!cir.array<!rec_NoOps x 25>>), !cir.ptr<!rec_NoOps>
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
// TODO: Add Init here.
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
// TODO: Add Init here.
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
// CHECK-NEXT: %[[BITCAST:.*]] = cir.cast(bitcast, %[[TL_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NoOps x 5> x 5> x 5>>), !cir.ptr<!cir.array<!rec_NoOps x 125>>
// CHECK-NEXT: %[[ARR_SIZE:.*]] = cir.const #cir.int<125> : !u64i
// CHECK-NEXT: %[[DECAY:.*]] = cir.cast(array_to_ptrdecay, %[[BITCAST]] : !cir.ptr<!cir.array<!rec_NoOps x 125>>), !cir.ptr<!rec_NoOps>
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

