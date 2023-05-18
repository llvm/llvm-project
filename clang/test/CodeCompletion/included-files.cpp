// RUN: rm -rf %t && mkdir %t && cp %s %t/main.cc && mkdir %t/a && mkdir %t/QtCore && mkdir %t/Headers %t/Some.framework %t/Some.framework/Headers
// RUN: touch %t/foo.h %t/foo.hxx %t/foo.cc %t/a/foosys %t/a/foosys.h %t/QtCore/foosys %t/QtCore/foo.h
// RUN: touch %t/Headers/foosys %t/Headers/foo.h %t/Some.framework/Headers/foosys %t/Some.framework/Headers/foo.h

// Quoted string shows header-ish files from CWD, and all from system.
#include "foo.h"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:6:13 %t/main.cc | FileCheck -check-prefix=CHECK-1 %s
// CHECK-1-NOT: foo.cc"
// CHECK-1: foo.h"
// CHECK-1: foo.hxx"
// CHECK-1: foosys"

// Quoted string with dir shows header-ish files in that subdir.
#include "a/foosys"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:14:13 %t/main.cc | FileCheck -check-prefix=CHECK-2 %s
// CHECK-2-NOT: foo.h"
// CHECK-2: foosys.h"
// CHECK-2-NOT: foosys"

// Angled shows headers from system dirs.
#include <foosys>
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:21:13 %t/main.cc | FileCheck -check-prefix=CHECK-3 %s
// CHECK-3-NOT: foo.cc>
// CHECK-3-NOT: foo.h>
// CHECK-3: foosys>

// With -I rather than -isystem, the header extension is required.
#include <foosys>
// RUN: %clang -fsyntax-only -I %t/a -Xclang -code-completion-at=%t/main.cc:28:13 %t/main.cc | FileCheck -check-prefix=CHECK-4 %s
// CHECK-4-NOT: foo.cc>
// CHECK-4-NOT: foo.h>
// CHECK-4-NOT: foosys>

// Backslash handling.
#include "a\foosys"
// RUN: %clang -fsyntax-only -isystem %t/a -Xclang -code-completion-at=%t/main.cc:35:13 %t/main.cc -fms-compatibility | FileCheck -check-prefix=CHECK-5 %s
// CHECK-5: foosys.h"

// Qt headers don't necessarily have extensions.
#include <foosys>
// RUN: %clang -fsyntax-only -I %t/QtCore -Xclang -code-completion-at=%t/main.cc:40:13 %t/main.cc -fms-compatibility | FileCheck -check-prefix=CHECK-6 %s
// CHECK-6-NOT: foo.cc>
// CHECK-6: foo.h>
// CHECK-6: foosys>

// If the include path directly points into a framework's Headers/ directory, we allow extension-less headers.
#include <foosys>
// RUN: %clang -fsyntax-only -I %t/Some.framework/Headers -Xclang -code-completion-at=%t/main.cc:47:13 %t/main.cc -fms-compatibility | FileCheck -check-prefix=CHECK-7 %s
// CHECK-7-NOT: foo.cc>
// CHECK-7: foo.h>
// CHECK-7: foosys>

// Simply naming a directory "Headers" is not enough to allow extension-less headers.
#include <foosys>
// RUN: %clang -fsyntax-only -I %t/Headers -Xclang -code-completion-at=%t/main.cc:54:13 %t/main.cc -fms-compatibility | FileCheck -check-prefix=CHECK-8 %s
// CHECK-8-NOT: foo.cc>
// CHECK-8: foo.h>
// CHECK-8-NOT: foosys>
