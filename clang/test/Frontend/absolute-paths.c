// RUN: %clang_cc1 -Wno-error=return-type -fsyntax-only -I %S/Inputs/SystemHeaderPrefix/.. %s 2>&1 | FileCheck -DROOT_ABSOLUTE=%s -check-prefix=NORMAL -check-prefix=CHECK %s
// RUN: %clang_cc1 -Wno-error=return-type -fsyntax-only -I %S/Inputs/SystemHeaderPrefix/.. -fdiagnostics-absolute-paths %s 2>&1 | FileCheck -DROOT_ABSOLUTE=%s -check-prefix=ABSOLUTE -check-prefix=CHECK %s

#include "absolute-paths-import.h"
// NORMAL: In file included from {{.*}}absolute-paths.c:4:
// NORMAL-NOT: In file included from [[ROOT_ABSOLUTE]]:4:
// ABSOLUTE: In file included from [[ROOT_ABSOLUTE]]:4:

#include "absolute-paths.h"

// Check that the bogus prefix we added is stripped out even if absolute paths
// are disabled.
// NORMAL-NOT: SystemHeaderPrefix
// ABSOLUTE-NOT: SystemHeaderPrefix
// CHECK: warning: non-void function does not return a value


// For files which don't exist, just print the filename.
#line 123 "non-existant.c"
int g(void) {}
// NORMAL: non-existant.c:123:14: warning: non-void function does not return a value
// ABSOLUTE: non-existant.c:123:14: warning: non-void function does not return a value
