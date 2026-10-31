!RUN: %flang -E %s 2>&1 | FileCheck %s
!RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: dir$
!CHECK-NOT: error:
!dir$ fixed
        continue
!dir$ free
        end
