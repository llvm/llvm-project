! RUN: %flang_fc1 -emit-hlfir -finit-local-zero -o - %s | FileCheck %s


!CHECK-LABEL: func.func @_QPuninitialized_integer() {
!CHECK: %[[ZERO_ALLOCA:.*]] = fir.alloca i32
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFuninitialized_integerEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_integerEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK: fir.store %[[ZERO]] to %[[ZERO_ALLOCA]] : !fir.ref<i32>
!CHECK: hlfir.assign %[[ZERO_ALLOCA]] to %[[X]]#0 : !fir.ref<i32>, !fir.ref<i32>
subroutine uninitialized_integer
  integer :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_real() {
!CHECK: %[[ZERO_ALLOCA:.*]] = fir.alloca f32
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFuninitialized_realEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_realEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: fir.store %[[ZERO]] to %[[ZERO_ALLOCA]] : !fir.ref<f32>
!CHECK: hlfir.assign %[[ZERO_ALLOCA]] to %[[X]]#0 : !fir.ref<f32>, !fir.ref<f32>
subroutine uninitialized_real
   real :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_logical() {
!CHECK: %[[ZERO_ALLOCA:.*]] = fir.alloca !fir.logical<4>
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFuninitialized_logicalEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_logicalEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK: %false = arith.constant false
!CHECK: %[[CVT:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK: fir.store %[[CVT]] to %[[ZERO_ALLOCA]] : !fir.ref<!fir.logical<4>>
!CHECK: hlfir.assign %[[ZERO_ALLOCA]] to %[[X]]#0 : !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>
subroutine uninitialized_logical
   logical :: x
end subroutine


!CHECK-LABEL: func.func @_QPuninitialized_complex() {
!CHECK: %[[ZERO_ALLOCA:.*]] = fir.alloca complex<f32>
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca complex<f32> {bindc_name = "x", uniq_name = "_QFuninitialized_complexEx"}
!CHECK: %[[X:.*]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFuninitialized_complexEx"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK: %[[CONST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: %[[UNDEF:.*]] = fir.undefined complex<f32>
!CHECK: %[[IDX0:.*]] = fir.insert_value %[[UNDEF]], %[[CONST]], [0 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: %[[IDX1:.*]] = fir.insert_value %[[IDX0]], %[[CONST]], [1 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: fir.store %[[IDX1]] to %[[ZERO_ALLOCA]] : !fir.ref<complex<f32>>
!CHECK: hlfir.assign %[[ZERO_ALLOCA]] to %[[X]]#0 : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>
subroutine uninitialized_complex
   complex :: x
end subroutine

!CHECK-LABEL: func.func @_QPuninitialized_character() {
!CHECK: %[[ZERO_ALLOCA:.*]] = fir.alloca !fir.char<1>
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[X_ALLOCA:.*]] = fir.alloca !fir.char<1> {bindc_name = "x", uniq_name = "_QFuninitialized_characterEx"}
!CHECK: %3:2 = hlfir.declare %[[X_ALLOCA]] typeparams %[[ONE]] {uniq_name = "_QFuninitialized_characterEx"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK: %[[CONST:.*]] = arith.constant 0 : i32
!CHECK: %[[CVT:.*]] = fir.convert %[[CONST]] : (i32) -> i8
!CHECK: %[[UNDEF:.*]] = fir.undefined !fir.char<1>
!CHECK: %[[VAL:.*]] = fir.insert_value %[[UNDEF]], %[[CVT]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
!CHECK: fir.store %[[VAL]] to %[[ZERO_ALLOCA]] : !fir.ref<!fir.char<1>>
!CHECK: hlfir.assign %[[ZERO_ALLOCA]] to %[[X]]#0 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
subroutine uninitialized_character
   character :: x
end subroutine
