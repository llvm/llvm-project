!RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenmp -fopenmp-version=52

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
