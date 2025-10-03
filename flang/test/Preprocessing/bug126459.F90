! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s 2>&1 | FileCheck %s
!CHECK: ndir=0
#define BLANKMACRO
BLANKMACRO !$ ndir=0
end
