// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

template<typename T>
void do_things(unsigned A, unsigned B) {
  T *OnePtr;
#pragma acc parallel private(OnePtr[A:B])
// CHECK: acc.private.recipe @privatization__Bcnt1__ZTSPi : !cir.ptr<!cir.ptr<!s32i>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!s32i>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(OnePtr[B])
  ;
#pragma acc parallel private(OnePtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPi : !cir.ptr<!cir.ptr<!s32i>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!s32i>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
  ;

  T **TwoPtr;
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
#pragma acc parallel private(TwoPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPi : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  T ***ThreePtr;
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
#pragma acc parallel private(ThreePtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPPi : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;


  T *ArrayOfPtr[5];
#pragma acc parallel private(ArrayOfPtr[B][A:B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_Pi : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ArrayOfPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtr[B][B])
  ;
#pragma acc parallel private(ArrayOfPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_Pi : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!s32i> x 5>, !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>, ["openacc.private.init"] {alignment = 16 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
  ;

  using TArrayTy = T[5];
  TArrayTy *PtrToArrays;
#pragma acc parallel private(PtrToArrays[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_i : !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrToArrays[B][A:B])
  ;
#pragma acc parallel private(PtrToArrays[A:B][A:B])
  ;
#pragma acc parallel private(PtrToArrays)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPA5_i : !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  T **ArrayOfPtrPtr[5];
#pragma acc parallel private(ArrayOfPtrPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSA5_PPi : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
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
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_PPi : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_PPi : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!s32i>> x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  TArrayTy **PtrPtrToArray;
#pragma acc parallel private(PtrPtrToArray[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPA5_i : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT:} 
  ;
#pragma acc parallel private(PtrPtrToArray[B][B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPA5_i : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray)
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPA5_i : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>>
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;

  using PtrTArrayTy = T*[5];
  PtrTArrayTy *PtrArrayPtr;

#pragma acc parallel private(PtrArrayPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPA5_Pi : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
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
// #pragma acc parallel private(PtrArrayPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_Pi : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
#pragma acc parallel private(PtrArrayPtr[B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr)
// CHECK: acc.private.recipe @privatization__ZTSPA5_Pi : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s32i> x 5>>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
  ;
}

void use(unsigned A, unsigned B) {
  do_things<int>(A, B);
}

