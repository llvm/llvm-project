! RUN: %flang_fc1 -fopenmp -fsyntax-only %s

! Check that using %re/%im inside 'parallel' doesn't cause syntax errors.
subroutine test_complex_re_im
  complex :: cc(4) = (1,2)
  integer :: i

  !$omp parallel do private(cc)
    do i = 1, 4
      print *, cc(i)%re, cc(i)%im
    end do
  !$omp end parallel do
end subroutine
