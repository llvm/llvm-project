! REQUIRES:  aarch64-registered-target

! RUN: %flang_fc1 -S -O1 %s -triple aarch64-linux-gnu -mllvm -debug-pass=Structure -o %t_O0 2>&1 | FileCheck %s
! RUN: %flang_fc1 -S -O2 %s -triple aarch64-linux-gnu -mllvm -debug-pass=Structure -o %t_O2 2>&1 | FileCheck %s
! RUN: %flang_fc1 -S -O3 %s -triple aarch64-linux-gnu -mllvm -debug-pass=Structure -o %t_O3 2>&1 | FileCheck %s
! RUN: FileCheck --input-file=%t_O0 --check-prefix=CALL %s
! RUN: FileCheck --input-file=%t_O2 --check-prefix=CALL %s
! RUN: FileCheck --input-file=%t_O3 --check-prefix=CALL %s

! CHECK: Target Library Information
! CHECK: Runtime Library Function Analysis
! CHECK: Library Function Lowering Analysis

! CALL: {{callq|bl}} memcpy
program memcpy_test
  implicit none
  integer, parameter :: n = 100
  real :: a(n), b(n)
  integer :: i

  ! Initialize array a
  do i = 1, n
    a(i) = real(i)
  end do

  ! Array assignment - this should generate memcpy
  b = a

  ! Use array b to prevent optimization
  print *, b(1), b(n)

end program memcpy_test

