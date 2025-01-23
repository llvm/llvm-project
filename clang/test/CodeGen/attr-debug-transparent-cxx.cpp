// RUN: %clang -gdwarf -emit-llvm -S %s -o - | FileCheck %s

void foo(void) {}

struct A {
  [[clang::debug_transparent()]]
  A() {
    foo();
  }

  [[clang::debug_transparent()]]
  ~A() {
    foo();
  }
[[clang::debug_transparent()]]
void method(void) {
  foo();
}

[[clang::always_inline()]]
[[clang::debug_transparent()]]
void inline_method(void) {
  foo();
}

};

int main() {
  auto a = A();
  a.method();
  a.inline_method();
}

// CHECK: DISubprogram(name: "inline_method"{{.*}} DISPFlagIsDebugTransparent
// CHECK: DISubprogram(name: "method"{{.*}} DISPFlagIsDebugTransparent
// CHECK: DISubprogram(name: "A"{{.*}} DISPFlagIsDebugTransparent
// CHECK: DISubprogram(name: "~A"{{.*}} DISPFlagIsDebugTransparent
