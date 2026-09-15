! RUN: not %flang_fc1 -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

! A local DECLARE REDUCTION shadows a USE-associated reduction
! (ProcessReductionSpecifier erases the USE-associated symbol). The shadowed
! type must not be resurrected through the operator's source module: the local
! reduction is authoritative because the operator is not merged.

module m_real_shadow
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

program test_shadowed_reduction
  use m_real_shadow   ! imports operator(.shared.) and the t_real reduction
  type :: t_int
    integer :: val = 0
  end type
  type(t_real) :: y
  integer :: i
  ! Local declaration shadows the USE-associated reduction (now only t_int).
  !$omp declare reduction(.shared.:t_int:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t_int(0))
  y = t_real(0.0)
  !CHECK: error: The type of 'y' is incompatible with the reduction operator.
  !$omp parallel do reduction(.shared.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp end parallel do
end program
