// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

__attribute__((overloadable)) void f1(a) int a; {
}
void f2(a) int a; {
}

// CHECK: !DISubprogram(name: "f1", linkageName: "_Z2f1i"
// CHECK: !DISubprogram(name: "f2", scope: {{.*}}, spFlags: DISPFlagDefinition,
