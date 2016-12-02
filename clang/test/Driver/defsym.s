// RUN: %clang -### -c -integrated-as %s \
// RUN: -Wa,-defsym,abc=5 -Wa,-defsym,xyz=0xa \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM1

// RUN: %clang -### -c -no-integrated-as -target x86_64-unknown-unknown %s \
// RUN: -Wa,-defsym,abc=5 -Wa,-defsym,xyz=0xa \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM1

// CHECK-DEFSYM1: "-defsym"
// CHECK-DEFSYM1: "abc=5"
// CHECK-DEFSYM1: "-defsym"
// CHECK-DEFSYM1: "xyz=0xa"

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym,abc= \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR1
// CHECK-DEFSYM-ERR1: error: defsym must be of the form: sym=value: abc=

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym,abc=1a2b3c \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR2
// CHECK-DEFSYM-ERR2: error: Value is not an integer: 1a2b3c
