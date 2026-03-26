// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style=LLVM -offset=2 -length=0 -offset=28 -length=0 \
// RUN:   | FileCheck -strict-whitespace %s
// CHECK: {{^int\ \*i;$}}
int*i;

// CHECK: {{^int\ \ \*\ \ i;$}}
int  *  i; 

// CHECK: {{^int\ \*i;$}}
int   *   i;
