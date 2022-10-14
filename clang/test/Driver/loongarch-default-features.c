// RUN: %clang --target=loongarch32 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang --target=loongarch64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64

// LA32: "target-features"="+d,+f"
// LA64: "target-features"="+d,+f"

/// Dummy function
int foo(void) {
  return  3;
}
