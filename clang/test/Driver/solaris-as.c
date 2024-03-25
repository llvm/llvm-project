/// General tests for assembler invocations on Solaris.

/// Test that clang uses gas on Solaris.
// RUN: %clang -x assembler %s -### -c -fno-integrated-as \
// RUN:         --target=sparc-sun-solaris2.11 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-GAS %s
// RUN: %clang -x assembler %s -### -c -fno-integrated-as \
// RUN:         --target=sparc-sun-solaris2.11 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-GAS %s
/// Allow for both "/usr/bin/gas" (native) and "gas" (cross) forms.
// CHECK-GAS: gas"
