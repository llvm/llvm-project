// REQUIRES: asserts
// RUN: %clang_cc1 -O0 -triple %itanium_abi_triple -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// Trigger GenerateVarArgsThunk.
// RUN: %clang_cc1 -O0 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s

// Check that retainedNodes are properly maintained at function cloning.
// RUN: %clang_cc1 -O1 -triple riscv64-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefixes=CHECK,CHECK-DI

// This test simply checks that the varargs thunk is created. The failing test
// case asserts.

struct Alpha {
  virtual void bravo(...);
};
struct Charlie {
  virtual ~Charlie() {}
};
struct CharlieImpl : Charlie, Alpha {
  void bravo(...) {}
} delta;

// CHECK: define {{.*}} void @_ZThn{{[48]}}_N11CharlieImpl5bravoEz(

// CHECK-DI: distinct !DISubprogram({{.*}}, linkageName: "_ZN11CharlieImpl5bravoEz", {{.*}}, retainedNodes: [[RN1:![0-9]+]]
// A non-empty retainedNodes list of original DISubprogram.
// CHECK-DI: [[RN1]] = !{!{{.*}}}

// CHECK-DI: distinct !DISubprogram({{.*}}, linkageName: "_ZN11CharlieImpl5bravoEz", {{.*}}, retainedNodes: [[EMPTY:![0-9]+]]
// An empty retainedNodes list of cloned DISubprogram.
// CHECK-DI: [[EMPTY]] = !{}
