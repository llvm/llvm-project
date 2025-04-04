! RUN: %flang -E -fopenmp %s 2>&1 | FileCheck %s
!CHECK: NDIR=0
#define BLANKMACRO
BLANKMACRO !$ NDIR=0
end
