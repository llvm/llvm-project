// RUN: %clang -### --target=x86_64 -fsplit-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK_OPT %s
// RUN: %clang -### --target=x86_64 -fprofile-use=default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK_NOOPT %s

// CHECK_OPT:        "-fsplit-machine-functions"
// CHECK_NOOPT-NOT:  "-fsplit-machine-functions"
