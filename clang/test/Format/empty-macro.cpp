// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format -style="{Macros: [A(x)=x]}" \
// RUN:   | FileCheck -strict-whitespace %s

// CHECK: A()
A()
