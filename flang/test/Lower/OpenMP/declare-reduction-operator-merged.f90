! Cross-module user-defined operator declare reduction via a merged generic. Two
! modules each declare operator(.remote.) and a single-type declare reduction:
! m_int for t_int, m_real for t_real. The program imports both with USE, ONLY,
! renaming operator(.remote.) to a common operator(.local.), so .local. is one
! merged generic; each reduction clause is disambiguated by the variable's type
! back to its own source module's reduction op (named from the source spelling
! op.remote., not the use-site rename op.local.). Each source declare is
! single-type; the multi-type-in-one-declare case (Form A) is covered by
! declare-reduction-operator-multiple-types.f90. Lowering counterpart of the
! semantics test Semantics/OpenMP/declare-reduction-use-only-merged.f90.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m_int
  type :: t_int
    integer :: v = 0
  end type
  interface operator(.remote.)
    module procedure add_int
  end interface
  !$omp declare reduction(.remote.:t_int:omp_out%v=omp_out%v+omp_in%v) &
  !$omp   initializer(omp_priv=t_int(0))
contains
  type(t_int) function add_int(a, b)
    type(t_int), intent(in) :: a, b
    add_int%v = a%v + b%v
  end function
end module

module m_real
  type :: t_real
    real :: v = 0.0
  end type
  interface operator(.remote.)
    module procedure add_real
  end interface
  !$omp declare reduction(.remote.:t_real:omp_out%v=omp_out%v+omp_in%v) &
  !$omp   initializer(omp_priv=t_real(0.0))
contains
  type(t_real) function add_real(a, b)
    type(t_real), intent(in) :: a, b
    add_real%v = a%v + b%v
  end function
end module

program main
  use m_int, only: t_int, operator(.local.) => operator(.remote.)
  use m_real, only: t_real, operator(.local.) => operator(.remote.)
  type(t_int) :: xi
  type(t_real) :: xr
  integer :: i
  xi = t_int(0)
  !$omp parallel do reduction(.local.:xi)
  do i = 1, 5
    xi%v = xi%v + i
  end do
  xr = t_real(0.0)
  !$omp parallel do reduction(.local.:xr)
  do i = 1, 4
    xr%v = xr%v + real(i)
  end do
  print '(I0,1X,F0.1)', xi%v, xr%v
end program

! Two source-scoped ops (m_int vs m_real owner), named from the source spelling
! op.remote. and disambiguated by type; the t_int loop is emitted first, then the
! t_real loop. Because the CHECK-DAG names require op.remote., an op wrongly keyed
! on the use-site rename op.local. would fail these checks.
! loose captures (R1) keyed on the owning module name; the two qualifiers differ.
! CHECK-DAG: omp.declare_reduction @[[REDINT:_QQ[A-Za-z0-9_.]*m_int[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK-DAG: omp.declare_reduction @[[REDREAL:_QQ[A-Za-z0-9_.]*m_real[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[REDINT]]
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[REDREAL]]
