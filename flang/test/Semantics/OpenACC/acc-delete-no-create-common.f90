! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! A named common block in the DELETE clause of EXIT DATA and in the
! NO_CREATE clause resolves like it does in the other data clauses, and
! NO_CREATE operands count as listed for DEFAULT(NONE).

module acc_delete_common_mod
  implicit none
  integer :: a
  common /c1/ a
contains
  subroutine module_procedure()
    !$acc enter data create(/c1/)
    a = 1
    !$acc exit data delete(/c1/)
    !$acc exit data finalize delete(/c1/)
  end subroutine
end module

subroutine external_subroutine()
  integer :: b
  common /c2/ b
  !$acc enter data create(/c2/)
  b = 1
  !$acc exit data delete(/c2/)
end subroutine

subroutine no_create_common()
  integer :: c
  common /c3/ c
  !$acc data no_create(/c3/)
  !$acc end data
  !$acc parallel no_create(/c3/)
  c = 1
  !$acc end parallel
end subroutine

subroutine default_none_no_create(n)
  integer :: n, i
  real :: x(n)
  !$acc parallel loop default(none) no_create(x) firstprivate(n)
  do i = 1, n
    x(i) = 1.0
  end do
end subroutine

subroutine missing_common()
  integer :: d
  common /c4/ d
  d = 1
!ERROR: Could not find COMMON block 'c5' used in OpenACC directive
  !$acc exit data delete(/c5/)
!ERROR: Could not find COMMON block 'c5' used in OpenACC directive
  !$acc data no_create(/c5/)
  !$acc end data
end subroutine

program main
  integer :: e
  common /c6/ e
  !$acc enter data create(/c6/)
  e = 1
  !$acc exit data delete(/c6/)
end program
