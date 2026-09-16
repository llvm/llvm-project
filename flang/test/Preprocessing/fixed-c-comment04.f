! Check that -E also enables preprocessing.
! RUN: %flang_fc1 -cpp -fsyntax-only %s
! RUN: %flang -E %s 2>&1 | FileCheck %s
! RUN: %flang -E -nocpp %s 2>&1 | FileCheck %s --check-prefix=NOCPP

! CHECK: print *, "hello", "world"
! NOCPP: /*c*/ print *, /* comment */ "hello"
/*c*/ print *, /* comment */ "hello"
/*d*/+, "world"
      end
