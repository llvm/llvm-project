! Separate-compilation renamed operator over a nested multi-type merge: three
! modules each declare reduction(.shared.:<type>:...) for a distinct type, an
! inner facade merges two under operator(.shared.), an outer facade merges the
! inner facade with the third, and the consumer imports the merged operator under
! a RENAMED spelling (operator(.local.) => operator(.shared.)) and reduces all
! three types with .local.. All three imported reductions must lower.
!
! The renamed operator is not visible under the reductions' source spelling
! (.shared.), so accessibility is decided by resolving from the local operator
! (.local.). A single merged operator names several reductions here, so the guard
! must collect every match (FindUserReductionSymbols), not just the first: with
! only the front match the int/real reductions would be dropped and their clauses
! would abort with "not yet implemented".
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_int.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_real.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_cplx.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_facade_inner.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_facade_outer.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 prog.f90 -o - | FileCheck prog.f90

!--- m_int.f90
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

!--- m_real.f90
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

!--- m_cplx.f90
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

!--- m_facade_inner.f90
! First level of merge: operator(.shared.) combined from m_int and m_real.
module m_facade_inner
  use m_int, only: t_int, operator(.shared.)
  use m_real, only: t_real, operator(.shared.)
end module

!--- m_facade_outer.f90
! Second level of merge: the inner facade's operator(.shared.) merged again with
! m_cplx's, so the int/real reductions are two levels deep from the consumer.
module m_facade_outer
  use m_facade_inner, only: t_int, t_real, operator(.shared.)
  use m_cplx, only: t_cplx, operator(.shared.)
end module

!--- prog.f90
! The consumer renames the merged operator on import. All three reductions are
! still materialized (named from their source modules) and every clause binds, so
! no TODO aborts the compile.
! CHECK-DAG: omp.declare_reduction @{{.*m_int.*op\.shared\..*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{.*m_real.*op\.shared\..*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{.*m_cplx.*op\.shared\..*}} : !fir.ref
! CHECK-NOT: not yet implemented
program test_rename_remerge
  use m_facade_outer, only: t_int, t_real, t_cplx, &
    operator(.local.) => operator(.shared.)
  type(t_int) :: x
  type(t_real) :: y
  type(t_cplx) :: z
  integer :: i
  x = t_int(0)
  y = t_real(0.0)
  z = t_cplx((0.0, 0.0))
  !$omp parallel do reduction(.local.:x)
  do i = 1, 10
    x%val = x%val + 1
  end do
  !$omp parallel do reduction(.local.:y)
  do i = 1, 10
    y%val = y%val + 1.0
  end do
  !$omp parallel do reduction(.local.:z)
  do i = 1, 10
    z%val = z%val + (1.0, 0.0)
  end do
end program
