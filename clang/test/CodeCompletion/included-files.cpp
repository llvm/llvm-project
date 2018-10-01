// RUN: rm -rf %t && mkdir %t && cp %s %t/main.cc && mkdir %t/a
// RUN: touch %t/foo.h && touch %t/foo.cc && touch %t/a/foosys %t/a/foosys.h

// Quoted string shows header-ish files from CWD, and all from system.
#include "foo.h"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:5:13 %t/main.cc | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1-NOT: foo.cc"
// CHECK-1: foo.h"
// CHECK-1: foosys"

// Quoted string with dir shows header-ish files in that subdir.
#include "a/foosys"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:12:13 %t/main.cc | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2-NOT: foo.h"
// CHECK-2: foosys.h"
// CHECK-2-NOT: foosys"

// Angled shows headers from system dirs.
#include <foosys>
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:19:13 %t/main.cc | FileCheck -check-prefix=CHECK-3 %s
// CHECK-3-NOT: foo.cc>
// CHECK-3-NOT: foo.h>
// CHECK-3: foosys>

// With -I rather than -isystem, the header extension is required.
#include <foosys>
// RUN: %clang -fsyntax-only -I %t/a -Xclang -code-completion-at=%t/main.cc:26:13 %t/main.cc | FileCheck -check-prefix=CHECK-4 %s
// CHECK-4-NOT: foo.cc>
// CHECK-4-NOT: foo.h>
// CHECK-4-NOT: foosys>

// Backslash handling.
#include "a\foosys"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:33:13 %t/main.cc -fms-compatibility | FileCheck -check-prefix=CHECK-5 %s
// CHECK-5: foosys.h"
