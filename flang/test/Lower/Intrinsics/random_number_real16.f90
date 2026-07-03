! REQUIRES: flang-supports-f128-math
! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_scalar
! CHECK: fir.call @_FortranARandomNumber16({{.*}}){{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> ()
subroutine test_scalar
  real(16) :: r
  call random_number(r)
end

! CHECK-LABEL: func @_QPtest_array
! CHECK: fir.call @_FortranARandomNumber16({{.*}}){{.*}}: (!fir.box<none>, !fir.ref<i8>, i32) -> ()
subroutine test_array(r)
  real(16) :: r(:)
  call random_number(r)
end
