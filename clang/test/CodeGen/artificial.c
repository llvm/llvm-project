// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

extern void foo();
// CHECK: !DISubprogram(name: "foo"
// CHECK-SAME: flags: DIFlagArtificial
inline void __attribute__((artificial)) foo() {}

void baz() {
  foo();
}
