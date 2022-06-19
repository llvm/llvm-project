       foo
// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -E -fminimize-whitespace %s | FileCheck -strict-whitespace %s --check-prefix=MINCOL
// RUN: %clang_cc1 -E -fminimize-whitespace -P %s | FileCheck -strict-whitespace %s --check-prefix=MINWS
       bar

// CHECK: {{^       }}foo
// CHECK: {{^       }}bar

// MINCOL: {{^}}foo
// MINCOL: {{^}}bar

// MINWS: {{^}}foo bar

