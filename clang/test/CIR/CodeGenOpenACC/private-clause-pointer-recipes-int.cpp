// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

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
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ThreePtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
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
// TODO: Add Init here.
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
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(TwoPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPi : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
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
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OnePtr[A:B])
  ;
}

void use(unsigned A, unsigned B) {
  do_things<int>(A, B);
}

