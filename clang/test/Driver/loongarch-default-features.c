// RUN: %clang --target=loongarch32 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang --target=loongarch64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64

// LA32-NOT: "target-features"=
// LA64: "target-features"="+64bit,+d,+f"

/// Dummy function
int foo(void) {
  return  3;
}
