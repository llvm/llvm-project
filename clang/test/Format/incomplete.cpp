// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format -style=LLVM -cursor=0 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{"IncompleteFormat": true, "Line": 2}}
// CHECK: {{^int\ \i;$}}
 int    i;
// CHECK: {{^f  \( g  \(;$}}
f  ( g  (;
