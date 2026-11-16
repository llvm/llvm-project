! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fsyntax-only %s

! Test that declare reduction resolves through a facade module that re-merges an
! already-merged operator with a further module's operator. This requires
! recursively traversing the merged generic's USE associations: the reductions
! for the first two types live in modules reached only through the inner facade.

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

module m_real
  type :: t_real
    real :: val = 0.0
  end type
  interface operator(.shared.)
    module procedure add_real
  end interface
  !$omp declare reduction(.shared.:t_real:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_real(0.0))
contains
  type(t_real) function add_real(a, b)
    type(t_real), intent(in) :: a, b
    add_real%val = a%val + b%val
  end function
end module

module m_cplx
  type :: t_cplx
    complex :: val = (0.0, 0.0)
  end type
  interface operator(.shared.)
    module procedure add_cplx
  end interface
  !$omp declare reduction(.shared.:t_cplx:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_cplx((0.0, 0.0)))
contains
  type(t_cplx) function add_cplx(a, b)
    type(t_cplx), intent(in) :: a, b
    add_cplx%val = a%val + b%val
  end function
end module

! Inner facade merges .shared. from m_int and m_real.
module m_facade_inner
  use m_int, only: t_int, operator(.shared.)
  use m_real, only: t_real, operator(.shared.)
end module

! Outer facade re-merges the inner (already-merged) operator with m_cplx.
module m_facade_outer
  use m_facade_inner, only: t_int, t_real, operator(.shared.)
  use m_cplx, only: t_cplx, operator(.shared.)
end module

program test_reexport_remerge_reduction
  use m_facade_outer
  type(t_int) :: x
  type(t_real) :: y
  type(t_cplx) :: z
  integer :: i
  x = t_int(0)
  y = t_real(0.0)
  z = t_cplx((0.0, 0.0))
  ! t_int and t_real reach only through the inner facade (recursive traversal).
  !$omp parallel do reduction(.shared.:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp end parallel do
  !$omp parallel do reduction(.shared.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp end parallel do
  !$omp parallel do reduction(.shared.:z)
  do i = 1, 10
    z%val = z%val + (1.0, 0.0)
  end do
  !$omp end parallel do
end program
