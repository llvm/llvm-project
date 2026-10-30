! RUN: split-file %s %t
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/continued.f90 2>&1 | FileCheck %s --check-prefix=CONTINUED --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/malformed.f90 2>&1 | FileCheck %s --check-prefix=MALFORMED
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/consecutive.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/trailing.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/fixed.f 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/external.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/after-external.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/after-program.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-program.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-unnamed-main.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-function.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-subroutine.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/submodule.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-function.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-subroutine.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-function.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-subroutine.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-procedure.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/interface.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/abstract-interface.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/generic-interface.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module-interface.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-module.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-submodule.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-module-procedure.f90 2>&1 | FileCheck %s --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-function.f90 2>&1 | FileCheck %s --implicit-check-not="error:"

! Three procedure forms in each context: untyped FUNCTION, typed FUNCTION,
! and SUBROUTINE. DECLARE is misplaced before each procedure header.
! Require exactly three errors at the directives, with no parser cascades.

! CHECK: error: Could not parse
! CHECK: error: OpenACC DECLARE directive must appear in a specification part
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: ^
! CHECK: error: OpenACC DECLARE directive must appear in a specification part
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: ^
! CHECK: error: OpenACC DECLARE directive must appear in a specification part
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: ^

! CONTINUED: error: Could not parse
! CONTINUED: error: OpenACC DECLARE directive must appear in a specification part
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: ^
! CONTINUED: error: OpenACC DECLARE directive must appear in a specification part
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: ^
! CONTINUED: error: OpenACC DECLARE directive must appear in a specification part
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: ^
! MALFORMED: error:

!--- external.f90
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine

!--- after-external.f90
subroutine previous()
end subroutine
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine

!--- after-program.f90
program previous
end program
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine

!--- internal-program.f90
program host
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end program

!--- internal-unnamed-main.f90
real :: host_x
host_x = 0.0
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end

!--- internal-function.f90
function host()
real :: host
host = 0.0
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end function

!--- internal-subroutine.f90
subroutine host()
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end subroutine

!--- module.f90
module host
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
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
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end submodule

!--- internal-module-function.f90
module host
contains
function outer()
real :: outer
outer = 0.0
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end function
end module

!--- internal-module-subroutine.f90
module host
contains
subroutine outer()
contains
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
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
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
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
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
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
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end procedure
end submodule

!--- interface.f90
module host
interface
!$acc declare create(x)
function untyped()
real :: untyped
end function

!$acc declare create(x)
real function typed()
end function

!$acc declare create(x)
subroutine process()
end subroutine
end interface
end module

!--- abstract-interface.f90
module host
abstract interface
!$acc declare create(x)
function untyped()
real :: untyped
end function

!$acc declare create(x)
real function typed()
end function

!$acc declare create(x)
subroutine process()
end subroutine
end interface
end module

!--- generic-interface.f90
module host
interface generic_0
!$acc declare create(x)
function untyped()
real :: untyped
end function
end interface
interface generic_1
!$acc declare create(x)
real function typed()
end function
end interface
interface generic_2
!$acc declare create(x)
subroutine process()
end subroutine
end interface
end module

!--- module-interface.f90
module host
interface
!$acc declare create(x)
module function untyped()
real :: untyped
end function

!$acc declare create(x)
module real function typed()
end function

!$acc declare create(x)
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
!$acc declare create(x)
module function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
module real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
module subroutine process()
real :: x
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
!$acc declare create(x)
module function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
module real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
module subroutine process()
real :: x
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
!$acc declare create(x)
module procedure untyped
real :: x
untyped = 1.0
end procedure

!$acc declare create(x)
module procedure typed
real :: x
typed = 1.0
end procedure

!$acc declare create(x)
module procedure process
real :: x
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
!$acc declare create(x)
function untyped()
real :: untyped
real :: x
untyped = 1.0
end function

!$acc declare create(x)
real function typed()
real :: x
typed = 1.0
end function

!$acc declare create(x)
subroutine process()
real :: x
end subroutine
end procedure
end submodule

!--- fixed.f
c$acc declare create(x)
      function untyped()
      real :: untyped
      real :: x
      untyped = 1.0
      end function

c$acc declare create(x)
      real function typed()
      real :: x
      typed = 1.0
      end function

c$acc declare create(x)
      subroutine process()
      real :: x
      end subroutine

!--- consecutive.f90
!$acc declare create(x)
!$acc declare create(x)
!$acc declare create(x)
subroutine process()
end subroutine

!--- trailing.f90
module first
contains
!$acc declare create(x)
end module
module second
contains
!$acc declare create(x)
end module
module third
interface
!$acc declare create(x)
end interface
end module

!--- continued.f90
!$acc declare &
!$acc& create(x)
function untyped()
real :: untyped
untyped = 1.0
end function
!$acc declare &
!$acc& create(x)
real function typed()
typed = 1.0
end function
!$acc declare &
!$acc& create(x)
subroutine process()
end subroutine

!--- malformed.f90
module malformed
contains
!$acc declare create(
subroutine process()
end subroutine
end module
