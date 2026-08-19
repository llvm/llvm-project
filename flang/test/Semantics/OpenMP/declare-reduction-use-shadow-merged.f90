! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

! A local DECLARE REDUCTION for an operator is treated as authoritative for that
! operator in its scope: it shadows reductions that are otherwise reachable
! through the (merged) operator for other types. Here module `host` declares its
! own `.shared.` reduction for `t_loc` and also merges in m_int's `.shared.`
! reduction for `t_int` via USE...ONLY of the operator. Using `.shared.` on a
! `t_int` object is rejected, because the local declaration is authoritative.
!
! This is a deliberate, conservative choice. The precise OpenMP semantics for a
! local declaration coexisting with a merged-in reduction for a different type
! are not clearly specified, so the lookup never accepts a reduction that a
! local declaration might be intended to shadow (it never over-accepts). A
! future refinement could instead keep merged-in reductions for other types
! reachable; this test documents the current behavior so a change is deliberate.

module m_int
  type :: t_int
    integer :: val = 0
  end type
  interface operator(.shared.)
    module procedure add_int
  end interface
  !$omp declare reduction(.shared.:t_int:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_int(0))
contains
  type(t_int) function add_int(a, b)
    type(t_int), intent(in) :: a, b
    add_int%val = a%val + b%val
  end function
end module

module host
  use m_int, only: t_int, operator(.shared.)   ! merge in the t_int reduction
  type :: t_loc
    real :: val = 0.0
  end type
  interface operator(.shared.)
    module procedure add_loc
  end interface
  !$omp declare reduction(.shared.:t_loc:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_loc(0.0))
contains
  type(t_loc) function add_loc(a, b)
    type(t_loc), intent(in) :: a, b
    add_loc%val = a%val + b%val
  end function
  subroutine s()
    type(t_int) :: x
    integer :: i
    x = t_int(0)
    !CHECK: error: The type of 'x' is incompatible with the reduction operator.
    !$omp parallel do reduction(.shared.:x)
    do i = 1, 10
      x%val = x%val + 1
    end do
    !$omp end parallel do
  end subroutine
end module
