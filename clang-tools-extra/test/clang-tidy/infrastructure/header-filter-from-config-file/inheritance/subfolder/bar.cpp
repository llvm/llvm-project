// UNSUPPORTED: system-windows
// RUN: pushd %S
// RUN: cd ..
// RUN: clang-tidy -checks=-*,google-explicit-constructor %s -- -I "." 2>&1 | FileCheck %s
// RUN: popd
#include "foo.h"
// CHECK-NOT: foo.h:1:12: warning: single-argument constructors must be marked explicit

#include "bar.h"
// CHECK: bar.h:1:13: warning: single-argument constructors must be marked explicit
