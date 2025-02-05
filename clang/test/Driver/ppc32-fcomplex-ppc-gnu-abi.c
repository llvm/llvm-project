// RUN: %clang -### -S --target=ppc32 -fcomplex-ppc-gnu-abi %s 2>&1

// RUN: %clang -target ppc32-unknown-unknown-gnu -### -S %s -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-GNU
// CHECK-GNU: "-fcomplex-ppc-gnu-abi"

// RUN: not %clang -### --target=ppc64 -fcomplex-ppc-gnu-abi %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ERROR

// RUN: not %clang -### --target=ppc32-unknown-unknown-coff -fcomplex-ppc-gnu-abi %s 2>&1 \
// RUN:     | FileCheck %s -check-prefix=CHECK-ERROR

// CHECK-ERROR: error: unsupported option '-fcomplex-ppc-gnu-abi' for target
