! Test lowering of an OpenMP REDUCTION clause that uses a locally-declared
! user-defined operator on a derived type (#204299). This exercises the by-ref
! reduction path and the unwrapped-type guard: the clause variable type is a
! reference type while the op stores the unwrapped reduction type.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine red_derived
  type :: t
    integer :: val = 0
  end type
  interface operator(.myop.)
    function add_t(a, b)
      import :: t
      type(t), intent(in) :: a, b
      type(t) :: add_t
    end function add_t
  end interface
  !$omp declare reduction(.myop.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.myop.:x)
  do i = 1, 100
    x = x .myop. t(1)
  end do
  !$omp end parallel do
end subroutine red_derived

! The op name must be module-scoped (a mangled "_QQ..." generated name), NOT the
! bare operator spelling. The directive and clause must reference the same name.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.myop\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK-NOT: omp.declare_reduction @op.myop.
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
