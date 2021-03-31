// Test version information.

// RUN: %clang -Wa,--version -c -fintegrated-as %s -o /dev/null \
// RUN:   | FileCheck --check-prefix=IAS %s
// IAS: clang version

// RUN: %clang -Wa,--version -c -fno-integrated-as %s -o /dev/null \
// RUN:   | FileCheck --check-prefix=GAS %s
// GAS-NOT: clang
