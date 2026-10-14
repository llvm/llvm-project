// RUN: %clang --target=i386-unknown-linux-gnu -ffixed-edi -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-FIXED-EDI < %t %s
// CHECK-FIXED-EDI: "-target-feature" "+reserve-edi"

// RUN: not %clang --target=x86_64-unknown-linux-gnu -ffixed-edi -### %s 2>&1 | FileCheck --check-prefix=CHECK-NO-X64-EDI %s
// CHECK-NO-X64-EDI: error: unsupported option '-ffixed-edi' for target 'x86_64-unknown-linux-gnu'
