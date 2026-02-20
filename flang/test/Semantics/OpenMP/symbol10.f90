! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! Test that when using a variable from a module within an OpenMP Construct, such as taskloop or parallel do,
! they are not marked as OmpPreDetermined. This ensures that it is correctly identified as a module variable,
! rather than having Host Association details.

!DEF: /test_module Module
module test_module
 !DEF: /test_module/n PUBLIC ObjectEntity INTEGER(4)
 integer n
end module
!DEF: /test_parallel_do (Subroutine) Subprogram
subroutine test_parallel_do
 !DEF: /test_parallel_do/i ObjectEntity INTEGER(4)
 integer i
!$omp parallel do
 !DEF: /test_parallel_do/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  block
   !REF: /test_module
   use :: test_module
   !DEF: /test_parallel_do/OtherConstruct1/n (OmpPrivate) HostAssoc INTEGER(4)
   do n=1,3
    !REF: /test_parallel_do/OtherConstruct1/i
    !REF: /test_parallel_do/OtherConstruct1/n
    print *, i, n
   end do
  end block
 end do
!$omp end parallel do
end subroutine

!DEF: /test_task (Subroutine) Subprogram
subroutine test_task
 !DEF: /test_task/i ObjectEntity INTEGER(4)
 integer i
!$omp task
 !DEF: /test_task/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  block
   !REF: /test_module
   use :: test_module
   !DEF: /test_task/OtherConstruct1/n (OmpPrivate) HostAssoc INTEGER(4)
   do n=1,3
    !REF: /test_task/OtherConstruct1/i
    !REF: /test_task/OtherConstruct1/n
    print *, i, n
   end do
  end block
 end do
!$omp end task
end subroutine

!DEF: /test_taskloop (Subroutine) Subprogram
subroutine test_taskloop
 !DEF: /test_taskloop/i ObjectEntity INTEGER(4)
 integer i
!$omp taskloop
 !DEF: /test_taskloop/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,5
  block
   !REF: /test_module
   use :: test_module
   !DEF: /test_taskloop/OtherConstruct1/n (OmpPrivate) HostAssoc INTEGER(4)
   do n=1,3
    !REF: /test_taskloop/OtherConstruct1/i
    !REF: /test_taskloop/OtherConstruct1/n
    print *, i, n
   end do
  end block
 end do
!$omp end taskloop
end subroutine
