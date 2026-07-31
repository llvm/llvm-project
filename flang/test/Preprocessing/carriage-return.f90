! Tests that a bare carriage return (0x0d) appearing in the interior of a
! source line -- i.e. not as part of a CRLF/CR line ending -- is treated as
! insignificant whitespace outside of character literals, while still being
! preserved verbatim inside character literals.
!
! The carriage returns are generated with printf so that no literal CR byte is
! committed into the repository.

! A mid-line CR before a free-form '&' continuation must compile cleanly.
! RUN: printf 'program p\n  integer :: i, j\n  common /blk/\r &\n    i, j\nend\n' > %t-free.f90
! RUN: %flang_fc1 -fsyntax-only %t-free.f90

! A mid-line CR in fixed-form source must compile cleanly.
! RUN: printf '      program p\n      integer i, j\n      common /blk/ i\r, j\n      end\n' > %t-fixed.f
! RUN: %flang_fc1 -fsyntax-only -ffixed-form %t-fixed.f

! A CR inside a character literal must be preserved: the literal stays three
! characters long and its middle byte remains 0x0d (decimal 13). The quote
! characters around the literal are emitted with printf's octal escape \047 so
! the whole format string stays single-quoted and portable to lit's internal
! shell on all platforms.
! RUN: printf 'module m\n  character(*), parameter :: s = \047a\rb\047\n  integer, parameter :: slen = len(s)\n  integer, parameter :: midval = iachar(s(2:2))\nend module\n' > %t-literal.f90
! RUN: %flang_fc1 -fdebug-unparse %t-literal.f90 | FileCheck %s
! CHECK: slen = 3_4
! CHECK: midval = 13_4
