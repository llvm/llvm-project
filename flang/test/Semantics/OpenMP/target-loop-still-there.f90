!RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=60 -Werror -Wno-experimental-option %s | FileCheck --allow-empty %s

!CHECK-NOT: deprecated
subroutine f00
  implicit none
  integer :: i
  !$omp target loop
  do i = 1, 10
  end do
end
