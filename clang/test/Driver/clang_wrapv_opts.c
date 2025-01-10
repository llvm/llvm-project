// RUN: %clang -### -S -fwrapv -fno-wrapv -fwrapv %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
// CHECK1: -fwrapv
//
// RUN: %clang -### -S -fwrapv-pointer -fno-wrapv-pointer -fwrapv-pointer %s 2>&1 | FileCheck -check-prefix=CHECK1-POINTER %s
// CHECK1-POINTER: -fwrapv-pointer
//
// RUN: %clang -### -S -fstrict-overflow -fno-strict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// CHECK2: -fwrapv{{.*}}-fwrapv-pointer
//
// RUN: %clang -### -S -fwrapv -fstrict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// CHECK3: -fwrapv
//
// RUN: %clang -### -S -fwrapv-pointer -fstrict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK3-POINTER %s
// CHECK3-POINTER: -fwrapv-pointer
//
// RUN: %clang -### -S -fno-wrapv -fno-strict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
// CHECK4-NOT: -fwrapv
// CHECK4: -fwrapv-pointer
// CHECK4-NOT: -fwrapv
//
// RUN: %clang -### -S -fno-wrapv-pointer -fno-strict-overflow %s 2>&1 | FileCheck -check-prefix=CHECK4-POINTER %s
// CHECK4-POINTER-NOT: -fwrapv-pointer
// CHECK4-POINTER: -fwrapv
// CHECK4-POINTER-NOT: -fwrapv-pointer
