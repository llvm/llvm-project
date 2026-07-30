// RUN: %clang -### -fno-assume-unique-vtables %s -S 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -fno-assume-unique-vtables -fassume-unique-vtables %s -S 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// CHECK-OPT: "-fno-assume-unique-vtables"
// CHECK-NOOPT-NOT: "-fno-assume-unique-vtables"
