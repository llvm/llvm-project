// Confirm that -gsplit-dwarf=DIR is passed to linker

// RUN: %clang -target x86_64-unknown-linux -### %s -flto=thin -gsplit-dwarf -o a.out 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-DWO-DIR-DEFAULT < %t %s
//
// CHECK-LINK-DWO-DIR-DEFAULT: "-plugin-opt=dwo_dir=a.out_dwo"
