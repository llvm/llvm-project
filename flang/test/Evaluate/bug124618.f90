! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: error:
real x
print *, char(48, kind=size([x])) ! folds down to 1
end
