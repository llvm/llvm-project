! RUN: rm -rf %t && split-file %s %t && mkdir %t/a %t/b %t/c
! RUN: %flang_fc1 -fsyntax-only -J%t/a %t/a.f90
! RUN: %flang_fc1 -fsyntax-only -J%t/b %t/b.f90
! RUN: %flang_fc1 -fsyntax-only -I%t/b -J%t/b %t/needs-b.f90
! RUN: not %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/a -I%t/b -J%t/c %t/user1.f90 2>&1 | FileCheck --check-prefix=USER1 %s
! RUN: not %flang_fc1 -fsyntax-only -I%t/a -I%t/b -J%t/c %t/user2.f90 2>&1 | FileCheck --check-prefix=USER2 %s
! RUN: not %flang_fc1 -fsyntax-only -Wno-homonymous-module -I%t/a -I%t/b -J%t/c %t/user1.f90 2>&1 | FileCheck --check-prefix=QUIET %s
! RUN: %flang_fc1 -fsyntax-only -I%t/b -J%t/b %t/exports-b.f90
! RUN: not %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/a -I%t/b -J%t/c %t/user3.f90 2>&1 | FileCheck --check-prefix=USER3 %s
! RUN: %flang_fc1 -fsyntax-only -I%t/a -I%t/b -J%t/c %t/user4.f90 2>&1 | FileCheck --allow-empty --check-prefix=USER4 %s
! RUN: not %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/b -J%t/c %t/user1.f90 2>&1 | FileCheck --check-prefix=ONLYB %s
! RUN: not %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/a -I%t/b -J%t/c %t/user5.f90 2>&1 | FileCheck --check-prefix=USER5 %s
! RUN: mkdir %t/stale && %flang_fc1 -fsyntax-only -J%t/stale %t/stale.f90
! RUN: %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/stale -I%t/b -J%t/c %t/user6.f90 2>&1 | FileCheck --check-prefix=USER6 %s
! RUN: %flang_fc1 -fsyntax-only -I%t/stale -I%t/b -J%t/c %t/user6.f90 2>&1 | FileCheck --allow-empty --check-prefix=DEFAULT6 %s
! RUN: not %flang_fc1 -fsyntax-only -Whomonymous-module-search-order -I%t/a -I%t/b -J%t/c %t/user7.f90 2>&1 | FileCheck --check-prefix=USER7 %s
! RUN: not %flang_fc1 -fsyntax-only -Wno-homonymous-module -I%t/a -I%t/b -J%t/c %t/user3.f90 2>&1 | FileCheck --check-prefix=QUIET %s
! RUN: not %flang_fc1 -fsyntax-only -I%t/a -I%t/b -J%t/c %t/user8.f90 2>&1 | FileCheck --check-prefix=USER8 %s
! RUN: %flang_fc1 -fsyntax-only -I%t/a -J%t/a %t/exports-a.f90
! RUN: %flang_fc1 -fsyntax-only -I%t/a -I%t/b -J%t/c %t/user9.f90 2>&1 | FileCheck --check-prefix=USER9 %s
! RUN: %flang_fc1 -fsyntax-only -I%t/b -I%t/a -J%t/c %t/user9.f90 2>&1 | FileCheck --check-prefix=USER9 %s

! Two distinct modules named "modfile90" exist in directories a and b; only
! a's has "x".  Modules "modfile90needsb" (in a contained subprogram) and
! "modfile90exportsb" (at module scope) depend on b's.  A USE of "modfile90"
! in the program being compiled resolves to b's version because of that
! dependence, even though a's version appears earlier in the search path
! (user1, user3) or has already been read (user2).  Warn so that the
! subsequent errors can be understood.  No warning is emitted for the USE
! within a module file (user4) or when there is no other module with the
! same name (ONLYB).  The search path is scanned only once per module and
! required checksum, so repeated USEs warn once (user5).  A stale copy of
! b's module that appears earlier in the search path is also reported, but
! the program compiles (user6).  The search path cases are reported only with
! -Whomonymous-module-search-order; by default only a module already in use
! is (DEFAULT6).  The same holds when the module was already
! read for a module file (user7).  A module defined in the source being
! compiled was not selected by a checksum and is not reported (user8).  When
! both modules have already been read for module files, the one not used is
! reported; here x comes from a's module through modfile90exportsa (user9).

!--- a.f90
module modfile90
  integer, parameter :: n = 1
  integer :: x
end module

!--- b.f90
module modfile90
  integer, parameter :: n = 1
end module

!--- needs-b.f90
module modfile90needsb
contains
  subroutine s
    use modfile90
  end subroutine
end module

!--- exports-b.f90
module modfile90exportsb
  use modfile90
end module

!--- user1.f90
subroutine user1
  use modfile90needsb
  use modfile90
  implicit none
  x = 1
end subroutine

!--- user2.f90
module user2
  use modfile90, only: n
  use modfile90needsb
contains
  subroutine t
    use modfile90, only: x
  end subroutine
end module

!--- user3.f90
subroutine user3
  use modfile90exportsb
  use modfile90
  implicit none
  x = n
end subroutine

!--- user4.f90
subroutine user4
  use modfile90exportsb
  implicit none
  integer :: k
  k = n
end subroutine

!--- user5.f90
subroutine user5a
  use modfile90needsb
  use modfile90
  implicit none
  x = 1
end subroutine
subroutine user5b
  use modfile90needsb
  use modfile90
  implicit none
  x = 2
end subroutine

!--- stale.f90
module modfile90
  integer, parameter :: n = 2
end module

!--- user6.f90
subroutine user6
  use modfile90needsb
  use modfile90
  implicit none
  integer :: k
  k = n
end subroutine

!--- user7.f90
subroutine user7a
  use modfile90exportsb
  use modfile90
  implicit none
  x = 1
end subroutine
subroutine user7b
  use modfile90exportsb
  use modfile90
  implicit none
  x = 2
end subroutine

!--- user8.f90
module modfile90
  integer, parameter :: n = 3
end module
subroutine user8
  use modfile90needsb
  use modfile90
  implicit none
  x = n
end subroutine

!--- exports-a.f90
module modfile90exportsa
  use modfile90
end module

!--- user9.f90
subroutine user9
  use modfile90exportsa
  use modfile90exportsb
  use modfile90
  implicit none
  x = n
end subroutine

!USER1: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90needsb' depends on it, but a different module file for 'modfile90' appears earlier in the module search path in '{{.*}}a{{/|\\}}modfile90.mod' [-Whomonymous-module-search-order]
!USER1: error: No explicit type declared for 'x'

!USER2: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90needsb' depends on it, but a different module 'modfile90' is already in use; a program must not have two modules with the same name [-Whomonymous-module]
!USER2: user2.f90:2:7: Module 'modfile90' already in use
!USER2: error: 'x' not found in module 'modfile90'

!QUIET-NOT: warning:
!QUIET: error: No explicit type declared for 'x'

!USER3: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90exportsb' depends on it, but a different module file for 'modfile90' appears earlier in the module search path in '{{.*}}a{{/|\\}}modfile90.mod' [-Whomonymous-module-search-order]
!USER3: error: No explicit type declared for 'x'

!USER4-NOT: warning:
!USER4-NOT: error:

!ONLYB-NOT: warning:
!ONLYB: error: No explicit type declared for 'x'

!USER5: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90needsb' depends on it, but a different module file for 'modfile90' appears earlier in the module search path in '{{.*}}a{{/|\\}}modfile90.mod' [-Whomonymous-module-search-order]
!USER5-NOT: warning:
!USER5: error: No explicit type declared for 'x'
!USER5-NOT: warning:
!USER5: error: No explicit type declared for 'x'
!USER5-NOT: warning:

!USER6: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90needsb' depends on it, but a different module file for 'modfile90' appears earlier in the module search path in '{{.*}}stale{{/|\\}}modfile90.mod' [-Whomonymous-module-search-order]
!USER6-NOT: error:

!USER7: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90exportsb' depends on it, but a different module file for 'modfile90' appears earlier in the module search path in '{{.*}}a{{/|\\}}modfile90.mod' [-Whomonymous-module-search-order]
!USER7-NOT: warning:
!USER7: error: No explicit type declared for 'x'
!USER7-NOT: warning:
!USER7: error: No explicit type declared for 'x'
!USER7-NOT: warning:

!USER8-NOT: warning:
!USER8: error: No explicit type declared for 'x'

!USER9: warning: Module 'modfile90' from '{{.*}}b{{/|\\}}modfile90.mod' is used because module 'modfile90exportsb' depends on it, but a different module 'modfile90' is already in use; a program must not have two modules with the same name [-Whomonymous-module]
!USER9: modfile90exportsa.mod:{{.*}}: Module 'modfile90' already in use
!USER9-NOT: error:

!DEFAULT6-NOT: warning:
!DEFAULT6-NOT: error:
