// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// Tests interaction of /Yc / /Yu with /showIncludes
// REQUIRES: x86-registered-target

#include "header3.h"

// When building the pch, header1.h (included by header2.h), header2.h (the pch
// input itself) and header3.h (included directly, above) should be printed.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Ycheader2.h /FIheader2.h /Fp%t.pch /c /Fo%t -- %s \
// RUN:   | FileCheck --strict-whitespace -check-prefix=CHECK-YC %s
// CHECK-YC: Note: including file: {{[^ ]*header2.h}}
// CHECK-YC: Note: including file:  {{[^ ]*header1.h}}
// CHECK-YC: Note: including file: {{[^ ]*header3.h}}

// When using the pch, only the direct include is printed.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Yuheader2.h /FIheader2.h /Fp%t.pch /c /Fo%t -- %s \
// RUN:   | FileCheck --strict-whitespace -check-prefix=CHECK-YU %s
// CHECK-YU-NOT: Note: including file: {{.*pch}}
// CHECK-YU-NOT: Note: including file: {{.*header1.h}}
// CHECK-YU-NOT: Note: including file: {{.*header2.h}}
// CHECK-YU: Note: including file: {{[^ ]*header3.h}}

// When not using pch at all, all the /FI files are printed.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /FIheader2.h /c /Fo%t -- %s \
// RUN:   | FileCheck --strict-whitespace -check-prefix=CHECK-FI %s
// CHECK-FI: Note: including file: {{[^ ]*header2.h}}
// CHECK-FI: Note: including file:  {{[^ ]*header1.h}}
// CHECK-FI: Note: including file: {{[^ ]*header3.h}}

// Also check that /FI arguments before the /Yc / /Yu flags are printed right.

// /FI flags before the /Yc arg should be printed, /FI flags after it shouldn't.
// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Ycheader2.h /FIheader0.h /FIheader2.h /FIheader4.h /Fp%t.pch /c /Fo%t -- %s \
// RUN:   | FileCheck --strict-whitespace -check-prefix=CHECK-YCFI %s
// CHECK-YCFI: Note: including file: {{[^ ]*header0.h}}
// CHECK-YCFI: Note: including file: {{[^ ]*header2.h}}
// CHECK-YCFI: Note: including file:  {{[^ ]*header1.h}}
// CHECK-YCFI: Note: including file: {{[^ ]*header4.h}}
// CHECK-YCFI: Note: including file: {{[^ ]*header3.h}}

// RUN: %clang_cl -Werror /showIncludes /I%S/Inputs /Yuheader2.h /FIheader0.h /FIheader2.h /FIheader4.h /Fp%t.pch /c /Fo%t -- %s \
// RUN:   | FileCheck --strict-whitespace -check-prefix=CHECK-YUFI %s
// CHECK-YUFI-NOT: Note: including file: {{.*pch}}
// CHECK-YUFI-NOT: Note: including file: {{.*header0.h}}
// CHECK-YUFI-NOT: Note: including file: {{.*header2.h}}
// CHECK-YUFI-NOT: Note: including file: {{.*header1.h}}
// CHECK-YUFI: Note: including file: {{[^ ]*header4.h}}
// CHECK-YUFI: Note: including file: {{[^ ]*header3.h}}
