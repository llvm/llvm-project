!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
logical b
a = 0.
!CHECK: error: expected '('
!CHECK-NOT: error: expected '%LOC'
b = .t.
c = 0.
end
