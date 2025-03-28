// RUN: clang-tidy -dump-config -system-headers=true | FileCheck -check-prefix=CHECK-CONFIG-SYSTEM-HEADERS %s
// RUN: clang-tidy -dump-config -system-headers=false | FileCheck -check-prefix=CHECK-CONFIG-NO-SYSTEM-HEADERS %s
// RUN: clang-tidy -config='SystemHeaders: true' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-SYSTEM-HEADERS %s
// RUN: clang-tidy -config='SystemHeaders: false' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-NO-SYSTEM-HEADERS %s

// RUN: clang-tidy -system-headers=true -config='SystemHeaders: true' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-SYSTEM-HEADERS %s
// RUN: clang-tidy -system-headers=true -config='SystemHeaders: false' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-SYSTEM-HEADERS %s
// RUN: clang-tidy -system-headers=false -config='SystemHeaders: true' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-NO-SYSTEM-HEADERS %s
// RUN: clang-tidy -system-headers=false -config='SystemHeaders: false' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-NO-SYSTEM-HEADERS %s

// RUN: clang-tidy -help | FileCheck -check-prefix=CHECK-OPT-PRESENT %s

// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -system-headers=true %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-SYSTEM-HEADERS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -system-headers=false %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-NO-SYSTEM-HEADERS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -config='SystemHeaders: true' %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-SYSTEM-HEADERS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -config='SystemHeaders: false' %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-NO-SYSTEM-HEADERS %s

#include <system_header.h>
// CHECK-SYSTEM-HEADERS: system_header.h:1:13: warning: single-argument constructors must be marked explicit
// CHECK-NO-SYSTEM-HEADERS-NOT: system_header.h:1:13: warning: single-argument constructors must be marked explicit

// CHECK-CONFIG-NO-SYSTEM-HEADERS: SystemHeaders: false
// CHECK-CONFIG-SYSTEM-HEADERS: SystemHeaders: true
// CHECK-OPT-PRESENT: --system-headers
