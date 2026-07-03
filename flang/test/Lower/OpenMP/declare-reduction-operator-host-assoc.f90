!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Test a user-defined operator reduction (reduction(.myadd.:x)) declared in a
! procedure that host-associates the operator from its enclosing module.

module m
  interface operator(.myadd.)
    procedure add_t
  end interface
  type t
    integer :: i
  end type
contains
  function add_t(x, y)
    type(t), intent(in) :: x, y
    type(t) :: add_t
    add_t%i = x%i + y%i
  end function
  subroutine s(r, a)
    type(t) :: r, a(10)
    integer :: k
!$omp declare reduction(.myadd.:t:omp_out=omp_out.myadd.omp_in) initializer(omp_priv=t(0))
    r = t(0)
!$omp parallel do reduction(.myadd.:r)
    do k=1,10
       r = r .myadd. a(k)
    end do
  end subroutine
end module

! The reduction is materialized with a name qualified by the declaring scope
! (module m, subroutine s), and the clause binds to it.

! CHECK: omp.declare_reduction @[[RED:_QQMmFsop.myadd.]] : !fir.ref<!fir.type<{{.*}}>>

! CHECK-LABEL: func.func @_QMmPs
! CHECK: omp.wsloop {{.*}}reduction(byref @[[RED]] %{{.*}} -> %{{.*}} : !fir.ref<!fir.type<{{.*}}>>)

