// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr10 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr10 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr9 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr9 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr8 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr8 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr7 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -mcpu=pwr7 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr10 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr10 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr9 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr9 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr8 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr8 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s

// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr7 \
// RUN:   -mcrbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// RUN: %clang -target powerpc-ibm-aix %s -### -mcpu=pwr7 \
// RUN:   -mno-crbits -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s


// CHECK-NOCRBITS: "-target-feature" "-crbits"
// CHECK-CRBITS: "-target-feature" "+crbits"


// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr9 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr9 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr9 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr7 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr7 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr7 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr10 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr10 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr10 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr9 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr9 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr9 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr8 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr8 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr8 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS

// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr7 -emit-llvm \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr7 -mcrbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-CRBITS
// RUN: %clang -target powerpc-ibm-aix -mcpu=pwr7 -mno-crbits \
// RUN:   -emit-llvm -S %s -o - | FileCheck %s --check-prefix=HAS-NOCRBITS


// HAS-CRBITS: main(
// HAS-CRBITS: attributes #0 = {
// HAS-CRBITS-SAME: +crbits
// HAS-NOCRBITS: main(
// HAS-NOCRBITS: attributes #0 = {
// HAS-NOCRBITS-SAME: -crbits

int main(int argc, char *argv[]) {
  return 0;
}
