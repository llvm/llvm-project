// REQUIRES: arm-registered-target

// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions %s -c -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -target x86_64 -fsplit-machine-functions %s -c -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// RUN: %clang -c -target arm-unknown-linux-gnueabi -fsplit-machine-functions %s -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s

// CHECK-OPT:       "-fsplit-machine-functions"
// CHECK-NOOPT-NOT: "-fsplit-machine-functions"
// CHECK-TRIPLE:    warning: -fsplit-machine-functions is not valid for arm
