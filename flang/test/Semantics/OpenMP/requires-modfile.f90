!RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenmp -fopenmp-version=60

module req
contains

! The requirements from the subprograms should be added to the module.
subroutine f00
  !$omp requires reverse_offload
end

subroutine f01
  !$omp requires atomic_default_mem_order(seq_cst)
end
end module

module user
! The requirements from module req should be propagated to this module.
use req
  ! This has no effect, and should not be emitted.
  !$omp requires unified_shared_memory(.false.)
end module

module fold
  integer, parameter :: x = 10
  integer, parameter :: y = 33
  ! Make sure we can fold this expression to "true".
  !$omp requires dynamic_allocators(x < y)
end module

!Expect: req.mod
!module req
!!$omp requires atomic_default_mem_order(seq_cst)
!!$omp requires reverse_offload
!contains
!subroutine f00()
!end
!subroutine f01()
!end
!end

!Expect: user.mod
!module user
!use req,only:f00
!use req,only:f01
!!$omp requires atomic_default_mem_order(seq_cst)
!!$omp requires reverse_offload
!end

!Expect: fold.mod
!module fold
!integer(4),parameter::x=10_4
!integer(4),parameter::y=33_4
!!$omp requires dynamic_allocators
!end
