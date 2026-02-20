// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, IndentAccessModifiers: true, BreakBeforeBraces: Whitesmiths}" \
// RUN:   | FileCheck -strict-whitespace %s

// CHECK: struct foo1
// CHECK-NEXT: {{^  {}}
// CHECK-NEXT: {{^    int i;}}
// CHECK-EMPTY:
// CHECK-NEXT: {{^  private:}}
// CHECK-NEXT: {{^    int j;}}
// CHECK: {{^  [}]}}
struct foo1 {
int i;
private:
int j; };
