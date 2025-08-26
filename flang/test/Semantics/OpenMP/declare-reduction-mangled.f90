! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

!! Test that the name mangling for min & max (also used for iand, ieor and ior).
module mymod
  type :: tt
     real r
  end type tt
contains
  function mymax(a, b)
    type(tt) :: a, b, mymax
    if (a%r > b%r) then
       mymax = a
    else
       mymax = b
    end if
  end function mymax
end module mymod

program omp_examples
!CHECK-LABEL: MainProgram scope: OMP_EXAMPLES
  use mymod
  implicit none
  integer, parameter :: n = 100
  integer :: i
  type(tt) :: values(n), big, small

  !$omp declare reduction(max:tt:omp_out = mymax(omp_out, omp_in)) initializer(omp_priv%r = 0)
  !$omp declare reduction(min:tt:omp_out%r = min(omp_out%r, omp_in%r)) initializer(omp_priv%r = 1)

!CHECK: min, ELEMENTAL, INTRINSIC, PURE (Function): ProcEntity
!CHECK: mymax (Function): Use from mymax in mymod
!CHECK: op.max: UserReductionDetails TYPE(tt)
!CHECK: op.min: UserReductionDetails TYPE(tt)

  big%r = 0
  !$omp parallel do reduction(max:big)
!CHECK: big (OmpReduction, OmpExplicit): HostAssoc
!CHECK: max, INTRINSIC: ProcEntity  
  do i = 1, n
     big = mymax(values(i), big)
  end do

  small%r = 1
  !$omp parallel do reduction(min:small)
!CHECK: small (OmpReduction, OmpExplicit): HostAssoc
  do i = 1, n
     small%r = min(values(i)%r, small%r)
  end do
  
  print *, "small=", small%r, " big=", big%r
end program omp_examples
