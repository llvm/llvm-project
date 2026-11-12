// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format -style=LLVM -cursor=6 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^\{ "Cursor": 3, }}
// CHECK: {{^int\ \i;$}}
 int    i;
