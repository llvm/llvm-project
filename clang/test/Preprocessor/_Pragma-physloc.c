// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// CHECK: {{^}}#pragma x y z{{$}}
// CHECK: {{^}}#pragma a b c{{$}}

_Pragma("x y z")
_Pragma("a b c")

