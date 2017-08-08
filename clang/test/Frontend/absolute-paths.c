// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs/SystemHeaderPrefix/.. %s 2>&1 | FileCheck -check-prefix=NORMAL -check-prefix=CHECK %s
// RUN: %clang_cc1 -fsyntax-only -I %S/Inputs/SystemHeaderPrefix/.. -fdiagnostics-absolute-paths %s 2>&1 | FileCheck -check-prefix=ABSOLUTE -check-prefix=CHECK %s

#include "absolute-paths.h"

// Check whether the diagnostic from the header above includes the dummy
// directory in the path.
// NORMAL: SystemHeaderPrefix
// ABSOLUTE-NOT: SystemHeaderPrefix
// CHECK: warning: control reaches end of non-void function


// For files which don't exist, just print the filename.
#line 123 "non-existant.c"
int g() {}
// NORMAL: non-existant.c:123:10: warning: control reaches end of non-void function
// ABSOLUTE: non-existant.c:123:10: warning: control reaches end of non-void function
