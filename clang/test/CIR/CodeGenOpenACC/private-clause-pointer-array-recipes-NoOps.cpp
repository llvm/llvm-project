// RUN: not %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct NoOps { int i = 0; };

// using PtrTArrayTy = T*[5];
// PtrTArrayTy *PtrArrayPtr;
// CHECK: acc.private.recipe @privatization__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>>, ["openacc.private.init"]
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// using PtrTArrayTy = T*[5];
// PtrTArrayTy *PtrArrayPtr;
// #pragma acc parallel private(PtrArrayPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// using PtrTArrayTy = T*[5];
// PtrTArrayTy *PtrArrayPtr;
// #pragma acc parallel private(PtrArrayPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPA5_P5NoOps : !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// using TArrayTy = T[5];
// TArrayTy **PtrPtrToArray;
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// using TArrayTy = T[5];
// TArrayTy **PtrPtrToArray;
// #pragma acc parallel private(PtrPtrToArray[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
// 
// using TArrayTy = T[5];
// TArrayTy **PtrPtrToArray;
// #pragma acc parallel private(PtrPtrToArray[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
// 
//
// T **ArrayOfPtrPtr[5];
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>, !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>>, ["openacc.private.init"] {alignment = 16 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// T **ArrayOfPtrPtr[5];
// #pragma acc parallel private(ArrayOfPtrPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// T **ArrayOfPtrPtr[5];
// #pragma acc parallel private(ArrayOfPtrPtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSA5_PP5NoOps : !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!cir.ptr<!rec_NoOps>> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// using TArrayTy = T[5];
// TArrayTy *PtrToArrays;
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.array<!rec_NoOps x 5>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// using TArrayTy = T[5];
// TArrayTy *PtrToArrays;
// #pragma acc parallel private(PtrToArrays[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPA5_5NoOps : !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.array<!rec_NoOps x 5>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// T *ArrayOfPtr[5];
// CHECK-NEXT: acc.private.recipe @privatization__ZTSA5_P5NoOps : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.array<!cir.ptr<!rec_NoOps> x 5>, !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>>, ["openacc.private.init"] {alignment = 16 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
//
// T *ArrayOfPtr[5];
// #pragma acc parallel private(ArrayOfPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSA5_P5NoOps : !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.array<!cir.ptr<!rec_NoOps> x 5>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// T ***ThreePtr;
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// T ***ThreePtr;
// #pragma acc parallel private(ThreePtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// T ***ThreePtr;
// #pragma acc parallel private(ThreePtr[B][B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt3__ZTSPPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND3:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// T **TwoPtr;
// CHECK-NEXT: acc.private.recipe @privatization__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!cir.ptr<!rec_NoOps>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>>, ["openacc.private.init"] {alignment = 8 : i64}
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
// T **TwoPtr;
// #pragma acc parallel private(TwoPtr[B][B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt2__ZTSPP5NoOps : !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> init {
// CHECK-NEXT: ^bb0(%[[ARG:.*]]: !cir.ptr<!cir.ptr<!cir.ptr<!rec_NoOps>>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}, %[[BOUND2:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//
//
// T *OnePtr;
// CHECK-NEXT: acc.private.recipe @privatization__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}):
// CHECK-NEXT: cir.alloca !cir.ptr<!rec_NoOps>, !cir.ptr<!cir.ptr<!rec_NoOps>>, ["openacc.private.init"] {alignment = 8 : i64} 
// CHECK-NEXT: acc.yield 
// CHECK-NEXT: } 
//
// T *OnePtr;
// #pragma acc parallel private(OnePtr[B])
// CHECK-NEXT: acc.private.recipe @privatization__Bcnt1__ZTSP5NoOps : !cir.ptr<!cir.ptr<!rec_NoOps>> init {
// CHECK-NEXT: ^bb0(%arg0: !cir.ptr<!cir.ptr<!rec_NoOps>> {{.*}}, %[[BOUND1:.*]]: !acc.data_bounds_ty {{.*}}):
// TODO: Add Init here.
// CHECK-NEXT: acc.yield
// CHECK-NEXT: }
//

template<typename T>
void do_things(unsigned A, unsigned B) {
  T *OnePtr;
#pragma acc parallel private(OnePtr[A:B])
  ;
#pragma acc parallel private(OnePtr[B])
  ;
#pragma acc parallel private(OnePtr)
  ;

  T **TwoPtr;
#pragma acc parallel private(TwoPtr[B][B])
  ;
#pragma acc parallel private(TwoPtr[B][A:B])
  ;
#pragma acc parallel private(TwoPtr[A:B][A:B])
  ;
#pragma acc parallel private(TwoPtr)
  ;

  T ***ThreePtr;
#pragma acc parallel private(ThreePtr[B][B][B])
  ;
#pragma acc parallel private(ThreePtr[B][B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr[B][B])
  ;
#pragma acc parallel private(ThreePtr[B][A:B])
  ;
#pragma acc parallel private(ThreePtr[A:B][A:B])
  ;
#pragma acc parallel private(ThreePtr)
  ;


  T *ArrayOfPtr[5];
#pragma acc parallel private(ArrayOfPtr[B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtr[B][B])
  ;
#pragma acc parallel private(ArrayOfPtr)
  ;

  using TArrayTy = T[5];
  TArrayTy *PtrToArrays;
#pragma acc parallel private(PtrToArrays[B][B])
  ;
#pragma acc parallel private(PtrToArrays[B][A:B])
  ;
#pragma acc parallel private(PtrToArrays[A:B][A:B])
  ;
#pragma acc parallel private(PtrToArrays)
  ;

  T **ArrayOfPtrPtr[5];
#pragma acc parallel private(ArrayOfPtrPtr[B][B][B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr[A:B][A:B])
  ;
#pragma acc parallel private(ArrayOfPtrPtr)
  ;

  TArrayTy **PtrPtrToArray;
#pragma acc parallel private(PtrPtrToArray[B][B][B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][B])
  ;
#pragma acc parallel private(PtrPtrToArray[B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray[A:B][A:B])
  ;
#pragma acc parallel private(PtrPtrToArray)
  ;

  using PtrTArrayTy = T*[5];
  PtrTArrayTy *PtrArrayPtr;

#pragma acc parallel private(PtrArrayPtr[B][B][B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[A:B][A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][B])
  ;
#pragma acc parallel private(PtrArrayPtr[B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr[A:B][A:B])
  ;
#pragma acc parallel private(PtrArrayPtr)
  ;
}

void use(unsigned A, unsigned B) {
  do_things<NoOps>(A, B);
}

