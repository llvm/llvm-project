! RUN: %flang_fc1 -emit-hlfir -finit-local-zero -o - %s | FileCheck %s

!CHECK-LABEL: func.func @_QPuninitialized_integer() {
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFuninitialized_integerEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_integerEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK: hlfir.assign %[[ZERO]] to %[[X]]#0 : i32, !fir.ref<i32>
subroutine uninitialized_integer
  integer :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_real() {
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFuninitialized_realEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_realEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: hlfir.assign %[[ZERO]] to %[[X]]#0 : f32, !fir.ref<f32>
subroutine uninitialized_real
   real :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_logical() {
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFuninitialized_logicalEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_logicalEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK: %false = arith.constant false
!CHECK: %[[CVT:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK: hlfir.assign %[[CVT]] to %[[X]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
subroutine uninitialized_logical
   logical :: x
end subroutine


!CHECK-LABEL: func.func @_QPuninitialized_complex() {
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca complex<f32> {bindc_name = "x", uniq_name = "_QFuninitialized_complexEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_complexEx"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: %[[UNDEF:.*]] = fir.undefined complex<f32>
!CHECK: %[[IDX0:.*]] = fir.insert_value %[[UNDEF]], %[[CONST]], [0 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: %[[IDX1:.*]] = fir.insert_value %[[IDX0]], %[[CONST]], [1 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: hlfir.assign %[[IDX1]] to %[[X]]#0 : complex<f32>, !fir.ref<complex<f32>>
subroutine uninitialized_complex
   complex :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_character() {
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca !fir.char<1> {bindc_name = "x", uniq_name = "_QFuninitialized_characterEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] typeparams %[[ONE]] {uniq_name = "_QFuninitialized_characterEx"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK: %[[ZERO:.*]] = arith.constant 0 : i8
!CHECK: %[[CVT:.*]] = fir.convert %[[ONE]] : (index) -> i64
!CHECK: %[[CONST:.*]] = arith.constant 1 : i64
!CHECK: %[[COUNT:.*]] = arith.muli %[[CVT]], %[[CONST]] : i64
!CHECK: %[[CVT:.*]] = fir.convert %[[X]]#0 : (!fir.ref<!fir.char<1>>) -> !llvm.ptr
!CHECK: "llvm.intr.memset"(%[[CVT]], %[[ZERO]], %[[COUNT]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
subroutine uninitialized_character
   character :: x
end subroutine
