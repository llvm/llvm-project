!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! copyprivate (allocatable) test

subroutine test()
  integer expected(4)
  integer, allocatable,save  :: zptr(:)
  integer omp_get_thread_num

!$omp threadprivate(zptr)
  allocate(zptr(4))
  expected(1) = 1
  expected(2) = 2
  expected(3) = 3
  expected(4) = 4

!$omp parallel num_threads(8)
 if (.not.allocated(zptr)) allocate(zptr(4))
 zptr = 0  !! Zero out the data in all threads
!$omp single
  zptr(1)=1 !! Restore the data, and copy to all threads (copypriv)
  zptr(2)=2
  zptr(3)=3
  zptr(4)=4
!$omp end single copyprivate(zptr)
  call check(expected, zptr, 4)
!$omp end parallel
  call check(expected, zptr, 4)
end subroutine

program p
  call omp_set_num_threads(2)
  call test()
end program
