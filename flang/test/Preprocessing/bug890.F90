! RUN: %flang -E %s 2>&1 | FileCheck %s
!CHECK: subroutine sub()
#define empty
subroutine sub ( &
  empty)
end
