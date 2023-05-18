// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, EmptyLineBeforeAccessModifier: LogicalBlock}" -lines=1:14 \
// RUN:   | clang-format -style="{BasedOnStyle: LLVM, EmptyLineBeforeAccessModifier: Never}" -lines=14:40 \
// RUN:   | FileCheck -strict-whitespace %s

// CHECK: int i
// CHECK-NEXT: {{^$}}
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct foo1 {
  int i;

private:
  int j;
};

// CHECK: struct bar1
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct bar1 {
private:
  int i;
  int j;
};

// CHECK: int i
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct foo2 {
  int i;

private:
  int j;
};

// CHECK: struct bar2
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct bar2 {
private:
  int i;
  int j;
};

// CHECK: int j
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct foo3 {
  int i;
  int j;

private:
};

// CHECK: struct bar3
// CHECK-NEXT: {{^private:$}}
// CHECK: }
struct bar3 {

private:
  int i;
  int j;
};
