// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -mcpu=pwr10 -mpcrel -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-PCREL %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -mcpu=pwr10 -mno-pcrel -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOPCREL %s
// CHECK-NOPCREL: "-target-feature" "-pcrel"
// CHECK-PCREL: "-target-feature" "+pcrel"

// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -emit-llvm -S %s -o - | grep "attributes.*+pcrelative-memops"
// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -mpcrel -emit-llvm -S %s -o - | grep "attributes.*+pcrelative-memops"
// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -mno-pcrel -emit-llvm -S %s -o - | grep "attributes.*\-pcrelative-memops"

int main(int argc, char *argv[]) {
  return 0;
}
