! RUN: %flang -fopenmp -E %s 2>&1 | FileCheck %s
!CHECK: !$ NDIR=0
program main
integer NDIR
#define BLANKMACRO

BLANKMACRO !$ NDIR=0
end program main
