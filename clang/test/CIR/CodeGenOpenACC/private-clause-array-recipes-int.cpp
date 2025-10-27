// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

template<typename T>
void do_things(unsigned A, unsigned B) {
  T OneArr[5];
#pragma acc parallel private(OneArr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OneArr[B])
  ;
#pragma acc parallel private(OneArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_i : !cir.ptr<!cir.array<!s32i x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!s32i x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!s32i x 5>, !cir.ptr<!cir.array<!s32i x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  T TwoArr[5][5];
#pragma acc parallel private(TwoArr[A:B][A:B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_i : !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(TwoArr[B][A:B])
  ;
#pragma acc parallel private(TwoArr[B][B])
  ;
#pragma acc parallel private(TwoArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_i : !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.array<!s32i x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!s32i x 5> x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;

  T ThreeArr[5][5][5];
#pragma acc parallel private(ThreeArr[B][B][B])
// CHECK-NEXT:acc.private.recipe @privatization__Bcnt3__ZTSA5_A5_A5_i : !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
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
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_A5_A5_i : !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// CHECK-NEXT: %[[TL_ALLOCA:.*]] = cir.alloca !cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 4 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
#pragma acc parallel private(ThreeArr[B][A:B])
  ;
#pragma acc parallel private(ThreeArr[A:B][A:B])
  ;
#pragma acc parallel private(ThreeArr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_A5_A5_i : !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>, !cir.ptr<!cir.array<!cir.array<!cir.array<!s32i x 5> x 5> x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}
  ;
}

void use(unsigned A, unsigned B) {
  do_things<int>(A, B);
}

