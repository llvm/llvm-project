! RUN: %flang_fc1 -fopenacc -Wportability -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not='COLLAPSE on DO CONCURRENT is non-standard'
! RUN: %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=DEFAULT --allow-empty

! Portability warning for COLLAPSE on a DO CONCURRENT: emitted under
! -Wportability/-pedantic, silent by default. A bare DO CONCURRENT never warns.

subroutine collapse_do_concurrent(a)
  integer :: a(10,10,10)
  integer :: i, j, k

  ! CHECK: warning: COLLAPSE on DO CONCURRENT is non-standard [-Wportability]
  !$acc parallel loop collapse(3)
  do concurrent (i=1:10, j=1:10, k=1:10)
    a(i,j,k) = 0
  end do

  !$acc parallel loop
  do concurrent (i=1:10, j=1:10, k=1:10)
    a(i,j,k) = 0
  end do
end subroutine

! Default run: no warning.
! DEFAULT-NOT: COLLAPSE on DO CONCURRENT is non-standard
