// RUN: clang-tidy -checks=-*,google-explicit-constructor %s 2>&1 | FileCheck %s
#include "foo.h"
// CHECK: foo.h:1:12: warning: single-argument constructors must be marked explicit
