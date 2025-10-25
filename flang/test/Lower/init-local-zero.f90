! RUN: %flang_fc1 -emit-fir -finit-local-zero -o - %s | FileCheck %s

!CHECK: %[[const:.*]] =  arith.constant 0 : i32
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFuninitialized_integerEx"}
!CHECK: %[[X_DECL:.*]] = fir.declare %[[X]] {uniq_name = "_QFuninitialized_integerEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
!CHECK: fir.store %[[const]] to %[[X_DECL]] : !fir.ref<i32>
subroutine uninitialized_integer
  integer :: x
end subroutine

!CHECK: %[[const:.*]] =  arith.constant 0.000000e+00 : f32
!CHECK: %[[X:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFuninitialized_realEx"}
!CHECK: %[[X_DECL:.*]] = fir.declare %[[X]] {uniq_name = "_QFuninitialized_realEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
!CHECK: fir.store %[[const]] to %[[X_DECL]] : !fir.ref<f32>
subroutine uninitialized_real
   real :: x
end subroutine

!CHECK: %false = arith.constant false
!CHECK: %[[X:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFuninitialized_logicalEx"}
!CHECK: %[[X_DECL:.*]] = fir.declare %[[X]] {uniq_name = "_QFuninitialized_logicalEx"} : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
!CHECK: %[[CVT:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK: fir.store %[[CVT]] to %[[X_DECL]] : !fir.ref<!fir.logical<4>>
subroutine uninitialized_logical
   logical :: x
end subroutine

!CHECK: %[[const:.*]] =  arith.constant 0.000000e+00 : f32
!CHECK: %[[X:.*]] = fir.alloca complex<f32> {bindc_name = "x", uniq_name = "_QFuninitialized_complexEx"}
!CHECK: %[[X_DECL:.*]] = fir.declare %[[X]] {uniq_name = "_QFuninitialized_complexEx"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
!CHECK: %[[undef:.*]] = fir.undefined complex<f32>
!CHECK: %[[REAL:.*]] = fir.insert_value %[[undef]], %[[const]], [0 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: %[[COMPLEX:.*]] = fir.insert_value %[[REAL]], %[[const]], [1 : index] : (complex<f32>, f32) -> complex<f32>
!CHECK: fir.store %[[COMPLEX]] to %[[X_DECL]] : !fir.ref<complex<f32>>
subroutine uninitialized_complex
   complex :: x
end subroutine

!CHECK: %[[X:.*]] = fir.alloca !fir.char<1> {bindc_name = "x", uniq_name = "_QFuninitialized_characterEx"}
!CHECK: %[[X_DECL:.*]] = fir.declare %[[X]]  typeparams %c1 {uniq_name = "_QFuninitialized_characterEx"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
!CHECK: %[[ADDR:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1>>
!CHECK: %[[FUNC_DECL:.*]] = fir.declare %[[ADDR]] {{.*}}
!CHECK: %[[LOAD:.*]] = fir.load %[[FUNC_DECL]]
!CHECK: fir.store %[[LOAD]] to %[[X_DECL]]
subroutine uninitialized_character
   character :: x
end subroutine

!CHECK: fir.global linkonce @{{.*}} constant : !fir.char<1> {
!CHECK: %[[VAL:.*]] = fir.string_lit "\00"(1) : !fir.char<1>
!CHECK: fir.has_value %[[VAL]] : !fir.char<1>
!CHECK: }
