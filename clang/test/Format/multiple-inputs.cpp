// RUN: cp %s %t-1.cpp
// RUN: cp %s %t-2.cpp
// RUN: clang-format -style=LLVM %t-1.cpp %t-2.cpp|FileCheck -strict-whitespace %s

// CHECK: {{^int\ \*i;}}
// CHECK: {{^int\ \*i;}}
 int   *  i  ;
