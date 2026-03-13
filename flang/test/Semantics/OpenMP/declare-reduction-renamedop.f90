! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

!! Test that we can "rename" an operator when using a module's operator.
module module1
!CHECK:  Module scope: module1 size=0
  implicit none
  type :: t1
     real :: value
  end type t1
  interface operator(.mul.)
     module procedure my_mul
  end interface operator(.mul.)
!CHECK: .mul., PUBLIC (Function): Generic DefinedOp procs: my_mul
!CHECK: my_mul, PUBLIC (Function): Subprogram result:TYPE(t1) r (TYPE(t1) x,TYPE(t1)
!CHECK: t1, PUBLIC: DerivedType components: value
contains
    function my_mul(x, y) result(r)
      type(t1), intent(in) :: x, y
      type(t1) :: r
      r%value = x%value * y%value
    end function my_mul
end module module1

program test_omp_reduction
!CHECK: MainProgram scope: TEST_OMP_REDUCTION
  use module1, only: t1, operator(.modmul.) => operator(.mul.)

!CHECK: .modmul. (Function): Use from .mul. in module1
  implicit none

  type(t1) :: result
  integer :: i
  !$omp declare reduction (.modmul. : t1 : omp_out = omp_out .modmul. omp_in) initializer(omp_priv = t1(1.0))
!CHECK: op.modmul.: UserReductionDetails TYPE(t1)
!CHECK: t1: Use from t1 in module1
!CHECK: OtherConstruct scope: size=8 alignment=4 sourceRange=0 bytes
!CHECK: omp_in size=4 offset=0: ObjectEntity type: TYPE(t1)
!CHECK: omp_out size=4 offset=4: ObjectEntity type: TYPE(t1)
!CHECK: OtherConstruct scope: size=8 alignment=4 sourceRange=0 bytes
!CHECK: omp_orig size=4 offset=0: ObjectEntity type: TYPE(t1)
!CHECK: omp_priv size=4 offset=4: ObjectEntity type: TYPE(t1)
  result = t1(1.0)
  !$omp parallel do reduction(.modmul.:result)
  do i = 1, 10
     result = result .modmul. t1(real(i))
  end do
  !$omp end parallel do
end program test_omp_reduction
