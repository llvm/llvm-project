! RUN: rm -rf %t && split-file %s %t && cd %t
! A module that uses another module's user-defined declare reduction re-exports
! it. A reduction named by an operator has an internal mangled symbol name
! ("op.remote.", "op.+") that is not valid Fortran, so it must not be written as
! a "use,only:" item (that module file could not be re-parsed and reading it
! crashed); it is recovered through the re-exported operator instead. This holds
! for both a defined operator (".remote.") and an intrinsic operator ("+") with a
! user "interface operator(+)", including when a consumer reaches the reduction
! through both the base and the facade. A reduction named by a plain identifier
! ("myred") has a valid name and is re-exported as a normal "use,only:" item.

! RUN: %flang_fc1 -fsyntax-only -fopenmp rx_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp rx_wrap.f90
! RUN: FileCheck --check-prefix=OPMOD --input-file=rx_wrap.mod %s
! RUN: %flang_fc1 -fsyntax-only -fopenmp rx_use.f90

! RUN: %flang_fc1 -fsyntax-only -fopenmp io_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp io_wrap.f90
! RUN: FileCheck --check-prefix=IOPMOD --input-file=io_wrap.mod %s
! RUN: %flang_fc1 -fsyntax-only -fopenmp io_use.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp io_dual.f90

! RUN: %flang_fc1 -fsyntax-only -fopenmp nm_base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp nm_wrap.f90
! RUN: FileCheck --check-prefix=NMMOD --input-file=nm_wrap.mod %s
! RUN: %flang_fc1 -fsyntax-only -fopenmp nm_use.f90

! The defined operator is re-exported; the mangled reduction symbol is not.
! OPMOD: use rx_base,only:operator(.remote.)
! OPMOD-NOT: only:op.remote.
! The intrinsic operator is re-exported; the mangled reduction symbol is not, and
! the directive is not re-emitted (no facade-owned duplicate reduction).
! IOPMOD: use io_base,only:operator(+)
! IOPMOD-NOT: only:op.+
! IOPMOD-NOT: DECLARE REDUCTION
! The plainly-named reduction is re-exported as a normal use item.
! NMMOD: use nm_base,only:myred

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
module rx_wrap
  use rx_base
end module

!--- rx_use.f90
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

!--- nm_use.f90
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

!--- io_base.f90
module io_base
  type :: t
    integer :: val = 0
  end type
  interface operator(+)
    module procedure add_t
  end interface
  !$omp declare reduction(+:t:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- io_wrap.f90
module io_wrap
  use io_base
end module

!--- io_use.f90
program main
  use io_wrap
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(+:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

!--- io_dual.f90
! Reaching the reduction through both the base and the facade must bind one
! reduction, not a facade-owned duplicate.
program main
  use io_base
  use io_wrap
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(+:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program
