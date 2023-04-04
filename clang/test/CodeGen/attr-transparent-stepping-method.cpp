// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s

void bar(void) {}

struct A {
[[clang::transparent_stepping()]]
void foo(void) {
  bar();
}
};

int main() {
  A().foo();
}

// CHECK: DISubprogram(name: "foo"{{.*}} DISPFlagIsTransparentStepping
