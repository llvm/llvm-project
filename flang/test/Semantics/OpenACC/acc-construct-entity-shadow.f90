! RUN: %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty --implicit-check-not=error:

! A construct entity -- a DO CONCURRENT/FORALL index-name, or a variable
! declared in a BLOCK construct -- lives in its own scope, distinct from a
! like-named variable in an enclosing scope. Referencing or defining such an
! entity inside an OpenACC region must not be treated as redefining an
! enclosing DO variable. Enabling OpenACC must not turn this legal code into
! a "Cannot redefine DO variable" error.

! DO CONCURRENT index-name shadowing an enclosing DO variable.
subroutine dc_index_shadow(a, b, n, m)
  integer, intent(in) :: n, m
  real, intent(inout) :: a(n,m), b(n,m)
  integer :: i, j

  do i = 1, m
    !$acc parallel loop gang vector collapse(2) independent
    do concurrent (j = 1:n, i = 1:m)
      a(i,j) = b(i,j)
    end do
  end do

  ! Also with a plain associated loop (no collapse).
  do i = 1, m
    !$acc loop
    do concurrent (i = 1:m)
      a(i,1) = b(i,1)
    end do
  end do
end subroutine

! BLOCK-local variable, inside a DO CONCURRENT, shadowing an enclosing DO
! variable and defined within the block.
subroutine block_in_do_concurrent(a, b, n, m)
  integer, intent(in) :: n, m
  real, intent(inout) :: a(n,m), b(n,m)
  integer :: i, ii, j

  do i = 1, m
    !$acc parallel loop gang vector collapse(2) independent
    do concurrent (j = 1:n, ii = 1:m)
      block
        integer :: i
        i = ii
        a(i,j) = b(i,j)
      end block
    end do
  end do
end subroutine

! BLOCK-local variable, inside a plain loop nest, shadowing an enclosing DO
! variable and defined within the block.
subroutine block_in_do_loop(array)
  integer :: i, j, array(1000)

  !$acc parallel loop gang vector
  do i = 1, 1000
    do j = 1, 1000
      block
        integer :: i
        i = j
        array(i) = 42
      end block
    end do
  end do
end subroutine
