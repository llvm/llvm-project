! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=50 -module-dir %t %t/m.f90
! RUN: cat %t/m.mod | FileCheck --ignore-case %s

!--- m.f90
module m
  implicit none
  type :: t
    integer :: x
  end type t
  !$omp declare mapper(mymap : t :: v) map(v%x)
end module m

!CHECK: !$OMP DECLARE MAPPER(mymap:t::v) MAP(v%x)
