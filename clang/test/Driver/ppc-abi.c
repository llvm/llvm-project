// Check passing PowerPC ABI options to the backend.

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1-qpx | FileCheck -check-prefix=CHECK-ELFv1-QPX %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mcpu=a2q | FileCheck -check-prefix=CHECK-ELFv1-QPX %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mcpu=a2 -mqpx | FileCheck -check-prefix=CHECK-ELFv1-QPX %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mcpu=a2q -mno-qpx | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2 %s

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv2 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=altivec | FileCheck -check-prefix=CHECK-ELFv2 %s

// CHECK-ELFv1: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-ELFv1: "-target-abi" "elfv1"
// CHECK-ELFv1-QPX: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-ELFv1-QPX: "-target-abi" "elfv1-qpx"
// CHECK-ELFv2: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-ELFv2: "-target-abi" "elfv2"

