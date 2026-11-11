// RUN: %clang -### -fhash-mangled-names %s -c 2>&1 | FileCheck --check-prefix=CHECK-OPT %s
// RUN: %clang -### -fhash-mangled-names -fno-hash-mangled-names %s -c 2>&1 | FileCheck --check-prefix=CHECK-NOOPT %s
// RUN: %clang -### %s -c 2>&1 | FileCheck --check-prefix=CHECK-NOOPT %s

// CHECK-OPT: "-fhash-mangled-names"
// CHECK-NOOPT-NOT: "-fhash-mangled-names"
