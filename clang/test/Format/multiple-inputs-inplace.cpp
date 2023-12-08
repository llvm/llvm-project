// RUN: cp %s %t-1.cpp
// RUN: cp %s %t-2.cpp
// RUN: clang-format -style=LLVM -i %t-1.cpp %t-2.cpp
// RUN: FileCheck -strict-whitespace -input-file=%t-1.cpp %s
// RUN: FileCheck -strict-whitespace -input-file=%t-2.cpp %s

// CHECK: {{^int\ \*i;}}
 int   *  i  ;
