#include "cycle.h"
#include "foo.h"

// RUN: env CINDEXTEST_KEEP_GOING=1 c-index-test -test-print-type -I%S/Inputs %s 2> %t.stderr.txt | FileCheck %s
// RUN: FileCheck -check-prefix CHECK-DIAG %s < %t.stderr.txt

// Verify that we don't stop preprocessing after an include cycle.
// CHECK: VarDecl=global_var:1:12 [type=int] [typekind=Int] [isPOD=1]

// CHECK-DIAG: cycle.h:1:10: error: #include nested too deeply
