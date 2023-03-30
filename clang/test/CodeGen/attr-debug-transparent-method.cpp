// RUN: %clang -gdwarf -emit-llvm -S %s -o - | FileCheck %s

void bar(void) {}

struct A {
[[clang::debug_transparent()]]
void foo(void) {
  bar();
}
};

int main() {
  A().foo();
}

// CHECK: DISubprogram(name: "foo"{{.*}} DISPFlagIsDebugTransparent
