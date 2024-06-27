// XFAIL: target={{.*}}-aix{{.*}}

// RUN: %clang -### -c -integrated-as %s \
// RUN: -Wa,-defsym,abc=5 -Wa,--defsym,xyz=0xa \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM1

// RUN: %clang -### -c -no-integrated-as -target x86_64-unknown-unknown %s \
// RUN: -Wa,-defsym,abc=5 -Wa,--defsym,xyz=0xa \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM1-GAS

// CHECK-DEFSYM1: "--defsym" "abc=5" "--defsym" "xyz=0xa"
// CHECK-DEFSYM1-GAS: "-defsym" "abc=5" "--defsym" "xyz=0xa"

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym,abc= \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR1
// CHECK-DEFSYM-ERR1: error: defsym must be of the form: sym=value: abc=

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym,=123 \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR2
// CHECK-DEFSYM-ERR2: error: defsym must be of the form: sym=value: =123

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym,abc=1a2b3c \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR3
// CHECK-DEFSYM-ERR3: error: value is not an integer: 1a2b3c

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR4

// RUN: not %clang -c -integrated-as -o /dev/null %s \
// RUN: -Wa,-defsym, \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEFSYM-ERR4

// CHECK-DEFSYM-ERR4: error: defsym must be of the form: sym=value: -defsym
