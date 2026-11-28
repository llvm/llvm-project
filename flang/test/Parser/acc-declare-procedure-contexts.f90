! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/keyword-assignments.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/leading-statement-function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/routine.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/scopes.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/fixed.f
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/external.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-external.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/after-program.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-program.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-unnamed-main.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-subroutine.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/module.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/submodule.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-subroutine.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-subroutine.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-procedure.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/interface.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/abstract-interface.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/generic-interface.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/module-interface.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-module.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-submodule.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/separate-module-procedure.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-function.f90

! Preserve procedure parsing in every host context. For definitions, DECLARE
! belongs inside the procedure specification part. Interface bodies test the
! procedure headers themselves, without asserting DECLARE semantics there.

!--- external.f90
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine

!--- after-external.f90
subroutine previous()
end subroutine
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine

!--- after-program.f90
program previous
end program
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine

!--- internal-program.f90
program host
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end program

!--- internal-unnamed-main.f90
real :: host_x
host_x = 0.0
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end

!--- internal-function.f90
function host()
real :: host
host = 0.0
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end function

!--- internal-subroutine.f90
subroutine host()
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end subroutine

!--- module.f90
module host
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end module

!--- submodule.f90
module ancestor
interface
module subroutine anchor()
end subroutine
end interface
end module
submodule(ancestor) host
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end submodule

!--- internal-module-function.f90
module host
contains
function outer()
real :: outer
outer = 0.0
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end function
end module

!--- internal-module-subroutine.f90
module host
contains
subroutine outer()
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end subroutine
end module

!--- internal-submodule-function.f90
module ancestor
interface
module subroutine anchor()
end subroutine
end interface
end module
submodule(ancestor) host
contains
function outer()
real :: outer
outer = 0.0
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end function
end submodule

!--- internal-submodule-subroutine.f90
module ancestor
interface
module subroutine anchor()
end subroutine
end interface
end module
submodule(ancestor) host
contains
subroutine outer()
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end subroutine
end submodule

!--- internal-separate-module-procedure.f90
module ancestor
interface
module subroutine outer()
end subroutine
end interface
end module
submodule(ancestor) host
contains
module procedure outer
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end procedure
end submodule

!--- interface.f90
module host
interface
function untyped()
real :: untyped
end function

real function typed()
end function

subroutine process()
end subroutine
end interface
end module

!--- abstract-interface.f90
module host
abstract interface
function untyped()
real :: untyped
end function

real function typed()
end function

subroutine process()
end subroutine
end interface
end module

!--- generic-interface.f90
module host
interface generic_0
function untyped()
real :: untyped
end function
end interface
interface generic_1
real function typed()
end function
end interface
interface generic_2
subroutine process()
end subroutine
end interface
end module

!--- module-interface.f90
module host
interface
module function untyped()
real :: untyped
end function

module real function typed()
end function

module subroutine process()
end subroutine
end interface
end module

!--- separate-in-module.f90
module host
interface
module function untyped()
real :: untyped
end function
module real function typed()
end function
module subroutine process()
end subroutine
end interface
contains
module function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

module real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

module subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end module

!--- separate-in-submodule.f90
module ancestor
interface
module function untyped()
real :: untyped
end function
module real function typed()
end function
module subroutine process()
end subroutine
end interface
end module
submodule(ancestor) host
contains
module function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

module real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

module subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end submodule

!--- separate-module-procedure.f90
module ancestor
interface
module function untyped()
real :: untyped
end function
module real function typed()
end function
module subroutine process()
end subroutine
end interface
end module
submodule(ancestor) host
contains
module procedure untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end procedure

module procedure typed
real :: x
!$acc declare create(x)
typed = 1.0
end procedure

module procedure process
real :: x
!$acc declare create(x)
end procedure
end submodule

!--- internal-separate-module-function.f90
module ancestor
interface
module real function outer()
end function
end interface
end module
submodule(ancestor) host
contains
module procedure outer
outer = 1.0
contains
function untyped()
real :: untyped
real :: x
!$acc declare create(x)
untyped = 1.0
end function

real function typed()
real :: x
!$acc declare create(x)
typed = 1.0
end function

subroutine process()
real :: x
!$acc declare create(x)
end subroutine
end procedure
end submodule

!--- fixed.f
      function untyped()
      real :: untyped
      real :: x
c$acc declare create(x)
      untyped = 1.0
      end function

      real function typed()
      real :: x
c$acc declare create(x)
      typed = 1.0
      end function

      subroutine process()
      real :: x
c$acc declare create(x)
      end subroutine

!--- scopes.f90
! A leading directive can begin the specification part of an unnamed main.
!$acc declare create(x)
real :: x
real :: function(10)
function(1) = 2.0
x = function(1)
block
  real :: local
  !$acc declare create(local)
  local = x
end block
end

module data_owner
  real :: a, b, c, d
  !$acc declare create(a)
  !$acc declare copyin(b)
  !$acc declare device_resident(c)
  !$acc declare link(d)
end module

subroutine local_clauses()
  real :: a, b, c, d, e, f, g
  !$acc declare create(a)
  !$acc declare copyin(b)
  !$acc declare copy(c)
  !$acc declare copyout(d)
  !$acc declare present(e)
  !$acc declare deviceptr(f)
  !$acc declare device_resident(g)
end subroutine

subroutine keyword_statement_function()
  real :: function, subroutine, x
  function(x) = x * 2.0
  subroutine(x) = x + 1.0
  print *, function(1.0), subroutine(1.0)
end subroutine

!--- keyword-assignments.f90
!$acc declare create(x)
function = 1.0
subroutine = 2.0
x = function + subroutine
end

!--- leading-statement-function.f90
!$acc declare create(x)
function(t) = t * 2.0
x = function(1.0)
end

!--- routine.f90
!$acc routine(external) seq
subroutine external()
end subroutine

module routine_hosts
interface
  !$acc routine(iface) seq
  subroutine iface()
  end subroutine
end interface
contains
  !$acc routine(contained) seq
  subroutine contained()
  end subroutine
end module
