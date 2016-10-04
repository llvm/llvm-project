// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=ieee -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=preserve-sign -v 2>&1 | FileCheck -check-prefix=CHECK-PS %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=positive-zero -v 2>&1 | FileCheck -check-prefix=CHECK-PZ %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=foo -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID %s

// CHECK-IEEE: "-fdenormal-fp-math=ieee"
// CHECK-PS: "-fdenormal-fp-math=preserve-sign"
// CHECK-PZ: "-fdenormal-fp-math=positive-zero"
// CHECK-INVALID: error: invalid value 'foo' in '-fdenormal-fp-math=foo'
