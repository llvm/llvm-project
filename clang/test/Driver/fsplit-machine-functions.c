// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -target x86_64 -fsplit-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT %s
// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s
// RUN: not %clang -c -target arm-unknown-linux -fsplit-machine-functions %s 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s

// CHECK-OPT:       "-fsplit-machine-functions"
// CHECK-NOOPT-NOT: "-fsplit-machine-functions"
// CHECK-TRIPLE:    error: unsupported option '-fsplit-machine-functions' for target
