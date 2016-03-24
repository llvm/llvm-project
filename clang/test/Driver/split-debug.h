// Check that we aren't splitting debug output for modules builds that don't produce object files.
//
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -fmodules -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -fmodules -emit-module -fmodules-embed-all-files -fno-implicit-modules -fno-implicit-module-maps -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// FIXME: This should fail using clang, except that the type of the output for
// an object output with modules is given as clang::driver::types::TY_PCH
// rather than TY_Object.
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -fmodules -fmodule-format=obj -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-NO-ACTIONS < %t %s
//
// CHECK-NO-ACTIONS-NOT: objcopy
