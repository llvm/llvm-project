// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -mcpu=pwr10 -mprefixed -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-PREFIXED %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -mcpu=pwr10 -mno-prefixed -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOPREFIXED %s
// CHECK-NOPREFIXED: "-target-feature" "-prefixed"
// CHECK-PREFIXED: "-target-feature" "+prefixed"

// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -emit-llvm -S %s -o - | grep "attributes.*+prefix-instrs"
// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -mprefixed -emit-llvm -S %s -o - | grep "attributes.*+prefix-instrs"
// RUN: %clang -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -mno-prefixed -emit-llvm -S %s -o - | grep "attributes.*\-prefix-instrs"

int main(int argc, char *argv[]) {
  return 0;
}
