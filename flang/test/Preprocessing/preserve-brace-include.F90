! RUN: %flang -E -I%S %s | FileCheck %s

#define empty ERR_NONEXISTENT
! CHECK-NOT: :5:
#include <empty.h>
