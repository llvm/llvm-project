! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp %s | FileCheck %s

! Verify that a USE-renamed array (s_ary => ary) is correctly associated in an
! OpenMP region under the alias name, not the original module name.
! Regression test for https://github.com/llvm/llvm-project/issues/185344

module use_rename_mod1
  implicit none
  real(8), allocatable :: ary(:,:,:)
end module

module use_rename_mod2
  implicit none
  real(8), allocatable :: ary(:,:,:,:,:)
end module

program test
  use use_rename_mod1, only: s_ary => ary
  use use_rename_mod2, only: ary
  implicit none
  integer(4) :: i

  !$omp parallel do
    do i = 1, 10
      s_ary(i,1,1) = ary(i,1,1,1,1)
    end do
  !$omp end parallel do
end program
! The OMP region must create a host-association for s_ary under its alias name.
! CHECK:  MainProgram scope: TEST
! CHECK:    OtherConstruct scope:
! CHECK:      ary (OmpShared): HostAssoc => ary
! CHECK:      i (OmpPrivate, OmpPreDetermined): HostAssoc => i
! CHECK:      s_ary (OmpShared): HostAssoc => ary
