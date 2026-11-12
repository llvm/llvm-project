// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format -style=LLVM \
// RUN:   | FileCheck -strict-whitespace %s

// CHECK: {{^int\ \*i;}}
 int   *  i  ;
