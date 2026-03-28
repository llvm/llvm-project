!RUN: %flang -fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
!CHECK-NOT: warning: Value of uninitialized local variable 'b' is used but never defined [-Wused-undefined-variable]
real :: a, b
pointer(p,a)
p = loc(b)
a = 2.0
print *, b
end
