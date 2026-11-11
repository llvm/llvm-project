! RUN: split-file %s %t
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/continued.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CONTINUED --implicit-check-not="error:" -DFILE=continued.f90 -DLINE1=1 -DLINE2=7 -DLINE3=12
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/malformed.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=MALFORMED --implicit-check-not="error:" --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/consecutive.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=consecutive.f90 -DLINE1=1 -DLINE2=2 -DLINE3=3
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/trailing.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=trailing.f90 -DLINE1=3 -DLINE2=7 -DLINE3=11
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/fixed.f 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=fixed.f -DLINE1=1 -DLINE2=8 -DLINE3=14
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/external.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=external.f90 -DLINE1=1 -DLINE2=7 -DLINE3=13
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/after-external.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=after-external.f90 -DLINE1=3 -DLINE2=9 -DLINE3=15
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/after-program.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=after-program.f90 -DLINE1=3 -DLINE2=9 -DLINE3=15
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-program.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-program.f90 -DLINE1=3 -DLINE2=9 -DLINE3=15
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-unnamed-main.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-unnamed-main.f90 -DLINE1=4 -DLINE2=10 -DLINE3=16
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-function.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-function.f90 -DLINE1=5 -DLINE2=11 -DLINE3=17
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-subroutine.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-subroutine.f90 -DLINE1=3 -DLINE2=9 -DLINE3=15
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=module.f90 -DLINE1=3 -DLINE2=9 -DLINE3=15
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/submodule.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=submodule.f90 -DLINE1=9 -DLINE2=15 -DLINE3=21
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-function.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-module-function.f90 -DLINE1=7 -DLINE2=13 -DLINE3=19
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-module-subroutine.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-module-subroutine.f90 -DLINE1=5 -DLINE2=11 -DLINE3=17
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-function.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-submodule-function.f90 -DLINE1=13 -DLINE2=19 -DLINE3=25
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-submodule-subroutine.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-submodule-subroutine.f90 -DLINE1=11 -DLINE2=17 -DLINE3=23
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-procedure.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-separate-module-procedure.f90 -DLINE1=11 -DLINE2=17 -DLINE3=23
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/interface.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=interface.f90 -DLINE1=3 -DLINE2=8 -DLINE3=12
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/abstract-interface.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=abstract-interface.f90 -DLINE1=3 -DLINE2=8 -DLINE3=12
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/generic-interface.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=generic-interface.f90 -DLINE1=3 -DLINE2=9 -DLINE3=14
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module-interface.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=module-interface.f90 -DLINE1=3 -DLINE2=8 -DLINE3=12
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-module.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=separate-in-module.f90 -DLINE1=12 -DLINE2=18 -DLINE3=24
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-in-submodule.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=separate-in-submodule.f90 -DLINE1=14 -DLINE2=20 -DLINE3=26
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/separate-module-procedure.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=separate-module-procedure.f90 -DLINE1=14 -DLINE2=20 -DLINE3=26
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/internal-separate-module-function.f90 2>&1 | FileCheck %s --strict-whitespace --implicit-check-not="error:" -DFILE=internal-separate-module-function.f90 -DLINE1=12 -DLINE2=18 -DLINE3=24

! Three procedure forms in each context: untyped FUNCTION, typed FUNCTION,
! and SUBROUTINE. DECLARE is misplaced before each procedure header.
! Require exactly three errors at the directives, with no parser cascades.

! CHECK: error: Could not parse
! CHECK: [[FILE]]:[[LINE1]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: {{^  \^$}}
! CHECK: [[FILE]]:[[LINE2]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: {{^  \^$}}
! CHECK: [[FILE]]:[[LINE3]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CHECK-NEXT: {{[!c*]}}$acc declare create(x)
! CHECK-NEXT: {{^  \^$}}

! CONTINUED: error: Could not parse
! CONTINUED: [[FILE]]:[[LINE1]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: {{^  \^$}}
! CONTINUED: [[FILE]]:[[LINE2]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: {{^  \^$}}
! CONTINUED: [[FILE]]:[[LINE3]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CONTINUED-NEXT: !$acc declare &
! CONTINUED-NEXT: {{^  \^$}}
! The unmatched parenthesis is a prescanner control; balanced malformed
! directives that reach parser recovery are covered in acc-declare-recovery.f90.
! MALFORMED: error: Could not scan
! MALFORMED: malformed.f90:3:21: error: Unmatched '('

!--- external.f90
!$acc declare create(x)
function untyped()
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
real :: untyped, x
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
