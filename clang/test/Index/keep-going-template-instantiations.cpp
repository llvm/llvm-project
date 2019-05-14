#include "missing.h"
#include <keep-going-template-instantiations.h>

// RUN: env CINDEXTEST_KEEP_GOING=1 c-index-test -test-load-source none -I%S/Inputs %s 2>&1 | FileCheck %s
// CHECK-NOT: error: expected class name
