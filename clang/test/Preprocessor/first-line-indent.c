       foo
// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s
       bar

// CHECK: {{^       }}foo
// CHECK: {{^       }}bar

