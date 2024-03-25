// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=ieee -v 2>&1 | FileCheck -check-prefix=CHECK-IEEE %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=preserve-sign -v 2>&1 | FileCheck -check-prefix=CHECK-PS %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=positive-zero -v 2>&1 | FileCheck -check-prefix=CHECK-PZ %s

// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=dynamic -v 2>&1 | FileCheck -check-prefix=CHECK-DYNAMIC %s

// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=ieee -fno-fast-math -v 2>&1 | FileCheck -check-prefix=CHECK-NO-UNSAFE %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=ieee -fno-unsafe-math-optimizations -v 2>&1 | FileCheck -check-prefix=CHECK-NO-UNSAFE %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=foo -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID0 %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=ieee,foo -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID1 %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=foo,ieee -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID2 %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -fdenormal-fp-math=foo,foo -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID3 %s

// IEEE is the implied default, and the flag is not passed.
// CHECK-IEEE-NOT: -fdenormal-fp-math=
// CHECK-PS: "-fdenormal-fp-math=preserve-sign,preserve-sign"
// CHECK-PZ: "-fdenormal-fp-math=positive-zero,positive-zero"
// CHECK-DYNAMIC: "-fdenormal-fp-math=dynamic,dynamic"


// CHECK-NO-UNSAFE-NOT: "-fdenormal-fp-math=ieee"
// CHECK-INVALID0: error: invalid value 'foo' in '-fdenormal-fp-math=foo'
// CHECK-INVALID1: error: invalid value 'ieee,foo' in '-fdenormal-fp-math=ieee,foo'
// CHECK-INVALID2: error: invalid value 'foo,ieee' in '-fdenormal-fp-math=foo,ieee'
// CHECK-INVALID3: error: invalid value 'foo,foo' in '-fdenormal-fp-math=foo,foo'
