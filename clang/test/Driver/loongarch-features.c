// RUN: %clang --target=loongarch32 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA32
// RUN: %clang --target=loongarch64 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=LA64

// LA32: "target-features"="+32bit"
// LA64: "target-features"="+64bit,+d,+f,+lsx,+relax,+ual"

// RUN: %clang --target=loongarch32-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT-LINUX
// RUN: %clang --target=loongarch64-linux -### %s -fsyntax-only 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT-LINUX

// DEFAULT-LINUX: "-funwind-tables=2"

int foo(void) {
  return 3;
}
