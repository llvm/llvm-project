! Separate-compilation intrinsic-operator user declare reductions covering all six
! intrinsic operators a user reduction may use: + * .and. .or. .eqv. .neqv. Each
! is declared over a derived type through an overloaded operator interface in a
! module compiled on its own to a .mod. A separate consumer imports the type and
! the operators and reduces one variable per operator. Each clause is the
! intrinsic-operator bind site: it resolves the imported user reduction under the
! operator's mangled name (op.+, op.AND, ...), materializes its op on demand, and
! binds it, so no clause hits a "not yet implemented" TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp m.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 use.f90 -o - | FileCheck use.f90

!--- m.f90
module m
  type :: t
    integer :: i = 0
  end type
  interface operator(+)
    module procedure add_t
  end interface
  interface operator(*)
    module procedure mul_t
  end interface
  interface operator(.and.)
    module procedure and_t
  end interface
  interface operator(.or.)
    module procedure or_t
  end interface
  interface operator(.eqv.)
    module procedure eqv_t
  end interface
  interface operator(.neqv.)
    module procedure neqv_t
  end interface
  !$omp declare reduction(+:t:omp_out=omp_out+omp_in) initializer(omp_priv=t(0))
  !$omp declare reduction(*:t:omp_out=omp_out*omp_in) initializer(omp_priv=t(1))
  !$omp declare reduction(.and.:t:omp_out=omp_out.and.omp_in) initializer(omp_priv=t(1))
  !$omp declare reduction(.or.:t:omp_out=omp_out.or.omp_in) initializer(omp_priv=t(0))
  !$omp declare reduction(.eqv.:t:omp_out=omp_out.eqv.omp_in) initializer(omp_priv=t(1))
  !$omp declare reduction(.neqv.:t:omp_out=omp_out.neqv.omp_in) initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%i = a%i + b%i
  end function
  type(t) function mul_t(a, b)
    type(t), intent(in) :: a, b
    mul_t%i = a%i * b%i
  end function
  type(t) function and_t(a, b)
    type(t), intent(in) :: a, b
    and_t%i = iand(a%i, b%i)
  end function
  type(t) function or_t(a, b)
    type(t), intent(in) :: a, b
    or_t%i = ior(a%i, b%i)
  end function
  type(t) function eqv_t(a, b)
    type(t), intent(in) :: a, b
    eqv_t%i = not(ieor(a%i, b%i))
  end function
  type(t) function neqv_t(a, b)
    type(t), intent(in) :: a, b
    neqv_t%i = ieor(a%i, b%i)
  end function
end module

!--- use.f90
! Each intrinsic operator's user reduction is materialized under its
! module-scoped name (getScopedUserReductionName, byte-identical on the clause
! side) and bound; no clause aborts the compile. The scoped name embeds the
! operator's mangled spelling (op.+, op.*, op.AND, op.OR, op.EQV, op.NEQV).
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*op\.\*[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*op\.AND[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*op\.OR[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*op\.EQV[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-DAG: omp.declare_reduction @{{_QQ[A-Za-z0-9_.]*op\.NEQV[A-Za-z0-9_.]*}} : !fir.ref
! CHECK-NOT: not yet implemented
program main
  use m, only: t, operator(+), operator(*), operator(.and.), &
               operator(.or.), operator(.eqv.), operator(.neqv.)
  type(t) :: xa, xm, xand, xor, xeqv, xneqv
  integer :: k
  xa = t(0)
  !$omp parallel do reduction(+:xa)
  do k = 1, 10
    xa = xa + t(k)
  end do
  xm = t(1)
  !$omp parallel do reduction(*:xm)
  do k = 1, 5
    xm = xm * t(k)
  end do
  xand = t(-1)
  !$omp parallel do reduction(.and.:xand)
  do k = 1, 5
    xand = xand .and. t(k)
  end do
  xor = t(0)
  !$omp parallel do reduction(.or.:xor)
  do k = 1, 5
    xor = xor .or. t(k)
  end do
  xeqv = t(-1)
  !$omp parallel do reduction(.eqv.:xeqv)
  do k = 1, 5
    xeqv = xeqv .eqv. t(k)
  end do
  xneqv = t(0)
  !$omp parallel do reduction(.neqv.:xneqv)
  do k = 1, 5
    xneqv = xneqv .neqv. t(k)
  end do
  print *, xa%i, xm%i, xand%i, xor%i, xeqv%i, xneqv%i
end program
