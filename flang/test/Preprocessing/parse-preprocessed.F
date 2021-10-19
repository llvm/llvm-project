! Verify that the output from `-E` is valid fixed-form source. See
! https://bugs.llvm.org/show_bug.cgi?id=50993.

! RUN: %flang_fc1 -E %s 2>&1 | %flang_fc1 -fsyntax-only -ffixed-form 2>&1 | FileCheck %s --allow-empty

! CHECK-NOT: error
! CHECK-NOT: warning

! https://bugs.llvm.org/show_bug.cgi?id=51219
! CHECK-NOT: Character in fixed-form label field must be a digit

      PROGRAM HELLO
        write(*, *), "hello, world!"
c     Some irrelevant comment that's only valid in fixed-form
      END PROGRAM
