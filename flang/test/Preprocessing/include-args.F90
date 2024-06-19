! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: call foo(3.14159)
call foo (&
#include "args.h"
)
end
