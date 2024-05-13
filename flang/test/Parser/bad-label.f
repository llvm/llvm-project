! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
!CHECK: portability: Label should be in the label field
      goto 1;  1 continue
      end
