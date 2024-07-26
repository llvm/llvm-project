// RUN: %clang --target=loongarch32 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang --target=loongarch64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64

// LA32: "target-features"="+32bit"
// LA64: "target-features"="+64bit,+d,+f,+lsx,+ual"

int foo(void) {
  return 3;
}
