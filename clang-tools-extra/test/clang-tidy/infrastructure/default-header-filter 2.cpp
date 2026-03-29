
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -header-filter='' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-EMPTY %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -header-filter='.*' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-EXPLICIT %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-NO-SYSTEM %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' --config='{}' -system-headers %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-WITH-SYSTEM %s

#include "header1.h"
// CHECK-DEFAULT: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK-EMPTY-NOT: header1.h:1:12: warning:
// CHECK-EXPLICIT: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK-NO-SYSTEM: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK-WITH-SYSTEM-DAG: header1.h:1:12: warning: single-argument constructors must be marked explicit

#include <system-header.h>
// CHECK-DEFAULT-NOT: system-header.h:1:12: warning:
// CHECK-EMPTY-NOT: system-header.h:1:12: warning:
// CHECK-EXPLICIT-NOT: system-header.h:1:12: warning:
// CHECK-NO-SYSTEM-NOT: system-header.h:1:12: warning:
// CHECK-WITH-SYSTEM-DAG: system-header.h:1:12: warning: single-argument constructors must be marked explicit

class A { A(int); };
// CHECK-DEFAULT: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit
// CHECK-EMPTY: :[[@LINE-2]]:11: warning: single-argument constructors must be marked explicit
// CHECK-EXPLICIT: :[[@LINE-3]]:11: warning: single-argument constructors must be marked explicit
// CHECK-NO-SYSTEM: :[[@LINE-4]]:11: warning: single-argument constructors must be marked explicit
// CHECK-WITH-SYSTEM: :[[@LINE-5]]:11: warning: single-argument constructors must be marked explicit
