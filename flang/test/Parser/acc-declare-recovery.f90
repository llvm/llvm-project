! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only %t/headers.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/assignment.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/main.f90
! RUN: %flang_fc1 -fsyntax-only %t/interposed.f90
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/external-malformed.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=ONE -DFILE=external-malformed.f90 -DLINE=1 -DCOL=1 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module-malformed.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=ONE -DFILE=module-malformed.f90 -DLINE=3 -DCOL=1 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/interface-malformed.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=ONE -DFILE=interface-malformed.f90 -DLINE=3 -DCOL=1 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/indented.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=INDENTED --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/interposed.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=INTERPOSED --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/consecutive-module.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CONSECUTIVE --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/trailing-internal.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=ONE -DFILE=trailing-internal.f90 -DLINE=5 -DCOL=1 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/fixed-continued.f 2>&1 | FileCheck %s --strict-whitespace --check-prefix=FIXED --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/fixed-ambiguous.f 2>&1 | FileCheck %s --strict-whitespace --check-prefix=AMBIGUOUS --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fopenacc -ffixed-form -fsyntax-only %t/fixed-cuda-ambiguous.cuf 2>&1 | FileCheck %s --strict-whitespace --check-prefix=AMBIGUOUS --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fsyntax-only %t/malformed-header.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=HEADER --implicit-check-not="error:"

! Check filenames and exact locations, as well as the diagnostic text. The
! malformed directives have balanced parentheses and reach parser recovery.
! ONE: error: Could not parse
! ONE: [[FILE]]:[[LINE]]:[[COL]]: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! ONE-NEXT: !$acc declare
! ONE-NEXT: {{^  \^$}}
! INDENTED: error: Could not parse
! INDENTED: indented.f90:3:5: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! INDENTED-NEXT:     !$acc declare create(x)
! INDENTED-NEXT: {{^      \^$}}
! INTERPOSED: error: Could not parse
! INTERPOSED: interposed.f90:1:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! INTERPOSED: interposed.f90:4:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CONSECUTIVE: error: Could not parse
! CONSECUTIVE: consecutive-module.f90:3:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! CONSECUTIVE: consecutive-module.f90:4:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! FIXED: error: Could not parse
! FIXED: fixed-continued.f:1:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! FIXED-NEXT: c$acc declare
! FIXED-NEXT: {{^  \^$}}

! In fixed form, a typed header with arguments is also a possible array
! declaration, and a CUDA header without arguments is also a possible
! ATTRIBUTES statement. Preserve declaration precedence and
! document this remaining diagnostic limitation rather than rejecting arrays.
! AMBIGUOUS-NOT: OpenACC DECLARE directive
! AMBIGUOUS: error: Could not parse
! AMBIGUOUS: in the context: END PROGRAM statement
! AMBIGUOUS-NOT: OpenACC DECLARE directive

! Normal SUBROUTINE parsing must retain its malformed-header recovery.
! HEADER: error: Could not parse
! HEADER: malformed-header.f90:1:14: error: expected end of statement

!--- headers.f90
recursive subroutine rec(a)
real :: a
end subroutine
subroutine empty()
end subroutine
subroutine noargs
end subroutine
subroutine alternate(*)
return 1
end subroutine
subroutine binding() bind(c)
end subroutine

!--- assignment.f90
!$acc declare create(subroutinex)
subroutinex = 1.0
end

!--- main.f90
!$acc declare create(x)
!$acc routine(s) seq
real :: x
external :: s
x = 1.0
call s()
end
subroutine s()
!$acc routine seq
end subroutine

!--- external-malformed.f90
!$acc declare create(x) bogus
subroutine s()
real :: x
end subroutine

!--- module-malformed.f90
module m
contains
!$acc declare bogus(x)
subroutine s()
real :: x
end subroutine
end module

!--- interface-malformed.f90
module m
interface
!$acc declare create(1)
subroutine s()
end subroutine
end interface
end module

!--- indented.f90
module m
contains
    !$acc declare create(x)
subroutine s()
real :: x
end subroutine
end module

!--- interposed.f90
!$acc declare create(x)
!$acc routine(s) seq
!dir$ optimize:1
!$acc declare create(y)
subroutine s()
real :: x, y
end subroutine

!--- consecutive-module.f90
module m
contains
!$acc declare create(x)
!$acc declare create(y)
subroutine s()
real :: x, y
end subroutine
end module

!--- trailing-internal.f90
program p
contains
subroutine s()
end subroutine
!$acc declare create(x)
end program

!--- fixed-continued.f
c$acc declare
c$acc+ create(x)
      subroutine s()
      real :: x
      end subroutine

!--- fixed-ambiguous.f
c$acc declare create(x)
      real function typed(a)
      real :: a, x
      typed = a
      end function

!--- malformed-header.f90
subroutine s bind(c)
end subroutine

!--- fixed-cuda-ambiguous.cuf
c$acc declare create(x)
      attributes(device) subroutine s
      real :: x
      end subroutine
