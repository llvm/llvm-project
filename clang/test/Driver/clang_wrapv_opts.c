// RUN: %clang -### -S -fwrapv -fno-wrapv -fwrapv %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
// CHECK1: -fwrapv

// RUN: %clang -### -S -fstrict-overflow -fno-strict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// CHECK2: -fwrapv

// RUN: %clang -### -S -fwrapv -fstrict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// CHECK3: -fwrapv

// RUN: %clang -### -S -fno-wrapv -fno-strict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
// CHECK4-NOT: -fwrapv
