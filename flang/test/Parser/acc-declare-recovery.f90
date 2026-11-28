! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only %t/headers.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/assignment.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/statement-function.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/main.f90
! RUN: %flang_fc1 -fopenacc -fsyntax-only %t/semicolon-main.f90
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
! RUN: not %flang_fc1 -fsyntax-only %t/malformed-args.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=ARGS --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/declare-malformed-args.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=RECOVERED -DFILE=declare-malformed-args.f90 -DCOL=13 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/declare-malformed-header.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=RECOVERED -DFILE=declare-malformed-header.f90 -DCOL=14 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/unsupported-wait.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=FALLBACK -DFILE=unsupported-wait.f90 --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/unsupported-routine.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=FALLBACK -DFILE=unsupported-routine.f90 --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fopenacc -fopenmp -fsyntax-only %t/unsupported-omp.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=FALLBACK -DFILE=unsupported-omp.f90 --implicit-check-not="OpenACC DECLARE"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/external-body-error.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=BODY -DFILE=external-body-error.f90 -DDECLARE=1 -DBODYLINE=3 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/module-body-error.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=BODY -DFILE=module-body-error.f90 -DDECLARE=3 -DBODYLINE=5 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/semicolon-subroutine.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=SEMICOLON -DFILE=semicolon-subroutine.f90 --implicit-check-not="error:"
! RUN: not %flang_fc1 -fopenacc -fsyntax-only %t/semicolon-function.f90 2>&1 | FileCheck %s --strict-whitespace --check-prefix=SEMICOLON -DFILE=semicolon-function.f90 --implicit-check-not="error:"

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
! AMBIGUOUS: error: Could not parse
! AMBIGUOUS: in the context: END PROGRAM statement

! Normal SUBROUTINE parsing must retain its malformed-header recovery.
! HEADER: error: Could not parse
! HEADER: malformed-header.f90:1:14: error: expected end of statement
! ARGS: error: Could not parse
! ARGS: malformed-args.f90:1:13: error: expected end of statement

! Recovered SUBROUTINE headers also identify misplaced DECLAREs. Keep the
! normal header diagnostic as well as the placement diagnostic.
! RECOVERED: error: Could not parse
! RECOVERED: [[FILE]]:1:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! RECOVERED: [[FILE]]:2:[[COL]]: error: expected end of statement

! Unsupported directives retain the normal diagnostic path instead of
! recovering a partial prefix and diagnosing its DECLARE in isolation.
! FALLBACK: error: Could not parse
! FALLBACK: [[FILE]]:3:12: error: expected '('

! Recovery must leave errors inside the following procedure visible.
! BODY: error: Could not parse
! BODY: [[FILE]]:[[DECLARE]]:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies
! BODY: [[FILE]]:[[BODYLINE]]:8: error: expected ':'
! BODY-NEXT: return ,
! BODY-NEXT: {{^         \^$}}

! A compiler directive may share a line with the following header. Lookahead
! and consumption must agree on where the directive ends.
! SEMICOLON: error: Could not parse
! SEMICOLON: [[FILE]]:1:1: error: OpenACC DECLARE directive must appear in the specification part of the scope to which it applies

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

!--- statement-function.f90
!$acc declare create(x)
subroutinex(a) = a
x = subroutinex(1.0)
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

!--- malformed-args.f90
subroutine s(a,)
end subroutine

!--- declare-malformed-args.f90
!$acc declare create(x)
subroutine s(a,)
end subroutine

!--- declare-malformed-header.f90
!$acc declare create(x)
subroutine s bind(c)
end subroutine

!--- unsupported-wait.f90
!$acc declare create(x)
!$acc wait
subroutine s()
end subroutine

!--- unsupported-routine.f90
!$acc declare create(x)
!$acc routine(s) seq junk
subroutine s()
end subroutine

!--- unsupported-omp.f90
!$acc declare create(x)
!$omp declare target
subroutine s()
end subroutine

!--- external-body-error.f90
!$acc declare create(x)
subroutine s()
return ,
end subroutine

!--- module-body-error.f90
module m
contains
!$acc declare create(x)
subroutine s()
return ,
end subroutine
end module

!--- semicolon-subroutine.f90
!$acc declare create(x)
!dir$ ivdep; subroutine s()
end subroutine

!--- semicolon-function.f90
!$acc declare create(x)
!dir$ ivdep; function f()
end function

!--- semicolon-main.f90
!$acc declare create(x)
!dir$ ivdep; real :: x
x = 1.0
end
