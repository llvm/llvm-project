// RUN: %clang -gdwarf -emit-llvm -S %s -o - | FileCheck %s

void bar(void) {}

__attribute__((debug_transparent))
void foo(void) {
  bar();
}

// CHECK: DISubprogram(name: "foo"{{.*}} DISPFlagIsDebugTransparent
