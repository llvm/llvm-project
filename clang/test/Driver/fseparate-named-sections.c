// RUN: %clang -### -fseparate-named-sections %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -fseparate-named-sections -fno-separate-named-sections %s -c 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// CHECK-OPT: "-fseparate-named-sections"
// CHECK-NOOPT-NOT: "-fseparate-named-sections"
