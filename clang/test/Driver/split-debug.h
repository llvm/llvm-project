// Check that we aren't splitting debug output for modules builds that don't produce object files.

// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -fmodules -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s

// CHECK-NO-ACTIONS-NOT: objcopy
