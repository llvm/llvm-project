// RUN: env CINDEXTEST_KEEP_GOING=1 c-index-test -test-load-source all %s > /dev/null 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-RANGE %s

#include <foobar.h>
#include "moozegnarf.h"

// CHECK-RANGE: rewrite-includes-missing.c:4:10:{4:10-4:19}: fatal error: 'foobar.h' file not found
// CHECK-RANGE: rewrite-includes-missing.c:5:10:{5:10-5:24}: fatal error: 'moozegnarf.h' file not found
