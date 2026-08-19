! Facade re-export: a module (rx_wrap) merely uses another module (rx_base) that
! declares an operator reduction, and a consumer uses the wrapper. rx_wrap must
! re-export the reduction so the consumer's clause binds. This also pins the
! module-file round-trip: the reduction's internal symbol (mangled "op.remote.")
! must not be written into rx_wrap.mod as an invalid use-only item that would
! make rx_wrap.mod fail to re-parse. The reduction is recovered through the
! re-exported operator.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp rx_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp rx_wrap.f90
! The wrapper module file must round-trip: no invalid "op.remote." use-only item.
! RUN: FileCheck --check-prefix=MODFILE --input-file=rx_wrap.mod rx_wrap.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp rx.use.f90 -o - | FileCheck rx.use.f90

! A plainly-named reduction ("myred") has a valid name, so it is re-exported as a
! normal use item and must still lower through the facade.
! RUN: %flang_fc1 -fsyntax-only -fopenmp nm_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp nm_wrap.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp nm.use.f90 -o - | FileCheck nm.use.f90

!--- rx_base.f90
module rx_base
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- rx_wrap.f90
! MODFILE: use rx_base,only:operator(.remote.)
! MODFILE-NOT: only:op.remote.
module rx_wrap
  use rx_base
end module

!--- rx.use.f90
! CHECK: omp.declare_reduction @[[RED:_QQMrx_baseop\.remote\._byref_rec__QMrx_baseTt]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use rx_wrap, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

!--- nm_base.f90
module nm_base
  type :: t
    integer :: val = 0
  end type
  !$omp declare reduction(myred:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
end module

!--- nm_wrap.f90
module nm_wrap
  use nm_base
end module

!--- nm.use.f90
! CHECK: omp.declare_reduction @[[RED:_QQMnm_basemyred_byref_rec__QMnm_baseTt]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use nm_wrap
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(myred:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program