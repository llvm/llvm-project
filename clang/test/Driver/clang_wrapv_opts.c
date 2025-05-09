// RUN: %clang -### -S -fwrapv -fno-wrapv -fwrapv -Werror %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
// CHECK1: "-fwrapv"

// RUN: %clang -### -S -fwrapv-pointer -fno-wrapv-pointer -fwrapv-pointer -Werror %s 2>&1 | FileCheck -check-prefix=CHECK1-POINTER %s
// CHECK1-POINTER: "-fwrapv-pointer"

// RUN: %clang -### -S -fstrict-overflow -fno-strict-overflow -Werror %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// CHECK2: "-fwrapv"{{.*}}"-fwrapv-pointer"

// RUN: %clang -### -S -fwrapv -fstrict-overflow -Werror -Werror %s 2>&1 | FileCheck -check-prefix=CHECK3 %s --implicit-check-not="-fwrapv-pointer"
// CHECK3-NOT: "-fwrapv"

// RUN: %clang -### -S -fwrapv-pointer -fstrict-overflow -Werror %s 2>&1 | FileCheck -check-prefix=CHECK3-POINTER %s --implicit-check-not="-fwrapv"
// CHECK3-POINTER-NOT: "-fwrapv-pointer"

// RUN: %clang -### -S -fno-wrapv -fno-strict-overflow -fno-wrapv-pointer -Werror %s 2>&1 | FileCheck -check-prefix=CHECK4 %s --implicit-check-not="-fwrapv-pointer"
// CHECK4: "-fwrapv"

// RUN: %clang -### -S -fno-wrapv-pointer -fno-strict-overflow -fno-wrapv -Werror %s 2>&1 | FileCheck -check-prefix=CHECK4-POINTER %s --implicit-check-not="-fwrapv"
// CHECK4-POINTER: "-fwrapv-pointer"
