// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// Tests interaction of /Yc / /Yu with /showIncludes

#include "header3.h"

// When building the pch, header1.h (included by header2.h), header2.h (the pch
// input itself) and header3.h (included directly, above) should be printed.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Ycheader2.h /FIheader2.h /Fp%t.pch /c -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// CHECK-YC: Note: including file: {{.+header2.h}}
// CHECK-YC: Note: including file: {{.+header1.h}}
// CHECK-YC: Note: including file: {{.+header3.h}}

// When using the pch, only the direct include is printed.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Yuheader2.h /FIheader2.h /Fp%t.pch /c -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU %s
// CHECK-YU-NOT: Note: including file: {{.*pch}}
// CHECK-YU-NOT: Note: including file: {{.*header1.h}}
// CHECK-YU-NOT: Note: including file: {{.*header2.h}}
// CHECK-YU: Note: including file: {{.+header3.h}}
