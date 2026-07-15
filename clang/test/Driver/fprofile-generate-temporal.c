// RUN: %clang -### -c -fprofile-generate -ftemporal-profile %s 2>&1 | FileCheck %s
// RUN: %clang -### -c -fcs-profile-generate -ftemporal-profile %s 2>&1 | FileCheck %s
// RUN: not %clang -### -c -ftemporal-profile %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK: "-mllvm" "--pgo-temporal-instrumentation"

// ERR: error: invalid argument '-ftemporal-profile' only allowed with '-fprofile-generate or -fcs-profile-generate'
