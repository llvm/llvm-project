
_Static_assert(0, "");

/// Test that piping the output into another process disables syntax
/// highlighting of code snippets.

// RUN: not %clang_cc1 %s -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: static assertion failed:
// CHECK-NEXT: {{^}}   2 | _Static_assert(0, "");{{$}}
