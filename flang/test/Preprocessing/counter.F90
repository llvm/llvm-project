! RUN: %flang -E %s | FileCheck %s
! CHECK: print *, 0
! CHECK: print *, 1
! CHECK: print *, 2
! Check incremental counter macro
#define foo bar
print *, __COUNTER__
print *, __COUNTER__
print *, __COUNTER__
