// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style=LLVM -lines=1:1 -lines=5:5 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^int\ \*i;$}}
  int*i;

// CHECK: {{^int\ \ \*\ \ i;$}}
int  *  i; 

// CHECK: {{^int\ \*i;$}}
int   *   i;

// RUN: not clang-format -lines=0:1 < %s 2>&1 \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK0 %s
// CHECK0: error: start line should be at least 1

// RUN: not clang-format -lines=2:1 < %s 2>&1 \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK1 %s
// CHECK1: error: start line should not exceed end line
