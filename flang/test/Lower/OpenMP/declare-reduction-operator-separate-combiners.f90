! Separate-compilation user-defined operator declare reductions whose combiner is
! not a plain intrinsic-assignment expression. These exercise the two typed forms
! (CallStmt::typedCall and AssignmentStmt::typedAssignment) that mod-file reading
! leaves null and the materializer must repopulate before lowering, in addition
! to the parser::Expr::typedExpr form covered by the plain cases. Both run to 100
! out of tree. https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t

! Case 1: the combiner is a subroutine call (combine_sub), so lowering reads the
! reduction combiner's CallStmt::typedCall. A null typedCall trips an assertion.
! RUN: %flang_fc1 -emit-hlfir -fopenmp call.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp call.use.f90 -o - | FileCheck call.use.f90

! Case 2: the combiner assigns a whole derived-type result through a defined
! assignment, so lowering reads AssignmentStmt::typedAssignment and takes the
! user-defined-assignment path.
! RUN: %flang_fc1 -emit-hlfir -fopenmp asgn.mod.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp asgn.use.f90 -o - | FileCheck asgn.use.f90

!--- call.mod.f90
module red_call
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:combine_sub(omp_out, omp_in)) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
  subroutine combine_sub(x, y)
    type(t), intent(inout) :: x
    type(t), intent(in) :: y
    x%val = x%val + y%val
  end subroutine
end module

!--- call.use.f90
! The combiner region must lower to a call of the imported subroutine.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: combiner
! CHECK: fir.call @_QMred_callPcombine_sub
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use red_call, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

!--- asgn.mod.f90
module red_asgn
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  interface assignment(=)
    module procedure assign_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out = add_t(omp_out, omp_in)) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
  subroutine assign_t(lhs, rhs)
    type(t), intent(out) :: lhs
    type(t), intent(in) :: rhs
    lhs%val = rhs%val
  end subroutine
end module

!--- asgn.use.f90
! The combiner region must lower a call to the imported defined assignment.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: combiner
! CHECK: fir.call @_QMred_asgnPassign_t
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use red_asgn, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program
