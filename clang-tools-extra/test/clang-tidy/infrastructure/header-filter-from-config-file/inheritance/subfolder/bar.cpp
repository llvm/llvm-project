// shell is required for the "dirname" command
// REQUIRES: shell
// RUN: clang-tidy -checks=-*,google-explicit-constructor %s -- -I "$(dirname %S)" 2>&1 | FileCheck %s
#include "foo.h"
// CHECK-NOT: foo.h:1:12: warning: single-argument constructors must be marked explicit

#include "bar.h"
// CHECK: bar.h:1:13: warning: single-argument constructors must be marked explicit
