// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoOps { int i = 0; };

// 'NoOps*', with 1 bound
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// 'NoOps*', no bounds
// CHECK-NEXT: acc.private.recipe @privatization__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}

// 'NoOps**', two bounds
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// 'NoOps**', 1 bounds
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// 'init' section:
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// 'NoOps**', no bounds
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}

// 'NoOps***', 3 bounds
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// 'NoOps***', 2 bounds
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// 'NoOps***', 1 bound
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }

// 'NoOps***', no bounds
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT:}

template<typename T>
void do_things(unsigned A, unsigned B) {

  T ***ThreePtr;
#pragma acc parallel private(ThreePtr)
  ;
#pragma acc parallel private(ThreePtr[A])
  ;
#pragma acc parallel private(ThreePtr[B][B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][B][B])
  ;
#pragma acc parallel private(ThreePtr[B][B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B][A:B])
  ;

  T **TwoPtr;
#pragma acc parallel private(TwoPtr)
  ;
#pragma acc parallel private(TwoPtr[A])
  ;
#pragma acc parallel private(TwoPtr[B][B])
  ;
#pragma acc parallel private(TwoPtr[B][A:B])
  ;
#pragma acc parallel private(TwoPtr[A:B][A:B])
  ;

  T *OnePtr;
#pragma acc parallel private(OnePtr)
  ;
#pragma acc parallel private(OnePtr[B])
  ;
#pragma acc parallel private(OnePtr[A:B])
  ;
}

void use(unsigned A, unsigned B) {
  do_things<NoOps>(A, B);
}

