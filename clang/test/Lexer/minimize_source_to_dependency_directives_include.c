// Test double slashes in #include directive along with angle brackets. Previously, this was interpreted as comments.
// RUN: %clang_cc1 -DTEST -print-dependency-directives-minimized-source %s 2>&1 | FileCheck %s

#include "a//b.h"
#include <a//b.h>

// CHECK: #include "a//b.h"
// CHECK: #include <a//b.h>
