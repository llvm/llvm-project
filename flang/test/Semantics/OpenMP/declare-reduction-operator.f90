! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

module m1
  interface operator(.fluffy.)
!CHECK: .fluffy., PUBLIC (Function): Generic DefinedOp procs: my_mul
    procedure my_mul
  end interface
  type t1
    integer :: val = 1
  end type
!$omp declare reduction(.fluffy.:t1:omp_out=omp_out.fluffy.omp_in)
!CHECK: op.fluffy., PUBLIC: UserReductionDetails TYPE(t1)
!CHECK: t1, PUBLIC: DerivedType components: val
!CHECK: OtherConstruct scope: size=16 alignment=4 sourceRange=0 bytes
!CHECK: omp_in size=4 offset=0: ObjectEntity type: TYPE(t1)
!CHECK: omp_orig size=4 offset=4: ObjectEntity type: TYPE(t1)
!CHECK: omp_out size=4 offset=8: ObjectEntity type: TYPE(t1)
!CHECK: omp_priv size=4 offset=12: ObjectEntity type: TYPE(t1)
contains
  function my_mul(x, y)
    type (t1), intent (in) :: x, y
    type (t1) :: my_mul
    my_mul%val = x%val * y%val
  end function my_mul

  subroutine subr(a, r)
    implicit none
    type(t1), intent(in), dimension(10) :: a
    type(t1), intent(out) :: r
    integer :: i
    !$omp parallel do reduction(.fluffy.:r)
    do i=1,10
       r = r .fluffy. a(i)
    end do
  end subroutine subr
end module m1
