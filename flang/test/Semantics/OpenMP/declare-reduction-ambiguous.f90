! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! A reduction identifier that resolves to more than one distinct user-defined
! reduction for the list item's type is ambiguous: the reduction to apply would
! otherwise be selected silently by USE order. This covers user-defined
! operators merged under one local name (by rename or by a shared name) and
! intrinsic-operator / special-function reduction names that collide across
! modules. https://github.com/llvm/llvm-project/issues/207255

! Two distinguishable defined operators renamed to one local operator, each with
! a reduction for the same type: ambiguous.
module ren_a
  interface operator(.aop.)
    module procedure ren_af
  end interface
  !$omp declare reduction(.aop.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
contains
  integer function ren_af(x, y)
    integer, intent(in) :: x, y
    ren_af = x + y
  end function
end module
module ren_b
  interface operator(.bop.)
    module procedure ren_bf
  end interface
  !$omp declare reduction(.bop.:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
contains
  real function ren_bf(x, y)
    real, intent(in) :: x, y
    ren_bf = x * y
  end function
end module
subroutine ren_sub
  use ren_a, operator(.local.) => operator(.aop.)
  use ren_b, operator(.local.) => operator(.bop.)
  integer :: x, i
  x = 0
  !ERROR: The reduction for 'x' is ambiguous: more than one user-defined reduction for its type is accessible.
  !$omp parallel do reduction(.local.:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine

! Two distinguishable defined operators that SHARE a name (no rename), each with
! a reduction for the same type: also ambiguous (also order-dependent).
module sam_a
  interface operator(.xop.)
    module procedure sam_af
  end interface
  !$omp declare reduction(.xop.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
contains
  integer function sam_af(x, y)
    integer, intent(in) :: x, y
    sam_af = x + y
  end function
end module
module sam_b
  interface operator(.xop.)
    module procedure sam_bf
  end interface
  !$omp declare reduction(.xop.:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
contains
  real function sam_bf(x, y)
    real, intent(in) :: x, y
    sam_bf = x * y
  end function
end module
subroutine sam_sub
  use sam_a
  use sam_b
  integer :: x, i
  x = 0
  !ERROR: The reduction for 'x' is ambiguous: more than one user-defined reduction for its type is accessible.
  !$omp parallel do reduction(.xop.:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine

! Two modules each declaring a reduction for the intrinsic operator + on the same
! type: the mangled name collides, ambiguous.
module int_a
  !$omp declare reduction(+:integer:omp_out=omp_out+2*omp_in) &
  !$omp   initializer(omp_priv=0)
end module
module int_b
  !$omp declare reduction(+:integer:omp_out=omp_out+3*omp_in) &
  !$omp   initializer(omp_priv=0)
end module
subroutine int_sub
  use int_a
  use int_b
  integer :: x, i
  x = 0
  !ERROR: The reduction for 'x' is ambiguous: more than one user-defined reduction for its type is accessible.
  !$omp parallel do reduction(+:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine

! Two modules each declaring a reduction for the special function max on the same
! type: ambiguous.
module spc_a
  !$omp declare reduction(max:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
end module
module spc_b
  !$omp declare reduction(max:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
end module
subroutine spc_sub
  use spc_a
  use spc_b
  integer :: x, i
  x = 1
  !ERROR: The reduction for 'x' is ambiguous: more than one user-defined reduction for its type is accessible.
  !$omp parallel do reduction(max:x)
  do i = 1, 5
    x = max(x, i)
  end do
  !$omp end parallel do
end subroutine

! Negative: a single renamed user-defined operator reduction is not ambiguous.
module one_a
  interface operator(.gop.)
    module procedure one_af
  end interface
  !$omp declare reduction(.gop.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
contains
  integer function one_af(x, y)
    integer, intent(in) :: x, y
    one_af = x + y
  end function
end module
subroutine one_sub
  use one_a, operator(.local.) => operator(.gop.)
  integer :: x, i
  x = 0
  !$omp parallel do reduction(.local.:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine

! Negative: two operators of DIFFERENT types (only one supports the integer list
! item) is not ambiguous for an integer reduction.
module typ_a
  interface operator(.top.)
    module procedure typ_af
  end interface
  !$omp declare reduction(.top.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
contains
  integer function typ_af(x, y)
    integer, intent(in) :: x, y
    typ_af = x + y
  end function
end module
module typ_b
  interface operator(.uop.)
    module procedure typ_bf
  end interface
  !$omp declare reduction(.uop.:real:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1.0)
contains
  real function typ_bf(x, y)
    real, intent(in) :: x, y
    typ_bf = x * y
  end function
end module
subroutine typ_sub
  use typ_a, operator(.local.) => operator(.top.)
  use typ_b, operator(.local.) => operator(.uop.)
  integer :: x, i
  x = 0
  !$omp parallel do reduction(.local.:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine

! Negative: a plain intrinsic reduction with no user-defined reduction in scope.
subroutine plain_sub
  integer :: x, i
  x = 0
  !$omp parallel do reduction(+:x)
  do i = 1, 5
    x = x + i
  end do
  !$omp end parallel do
end subroutine
