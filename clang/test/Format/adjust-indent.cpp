// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format -style=LLVM -lines=2:2 \
// RUN:   | FileCheck -strict-whitespace %s

void  f() {
// CHECK: void f() {
int i;
// CHECK: {{^  int\ i;}}
 int j;
// CHECK: {{^  int\ j;}}
}
