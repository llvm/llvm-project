// Confirm that -gsplit-dwarf=DIR is passed to linker

// RUN: %clang --target=x86_64-unknown-linux -### %s -flto=thin -gsplit-dwarf -o a.out 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-ELF-DWO-DIR-DEFAULT < %t %s
// RUN: %clang_cl --target=x86_64-unknown-windows-msvc -### -fuse-ld=lld -flto -gsplit-dwarf -o a.out -- %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-LINK-COFF-DWO-DIR-DEFAULT < %t %s
//
// CHECK-LINK-ELF-DWO-DIR-DEFAULT:  "-plugin-opt=dwo_dir=a.out_dwo"
// CHECK-LINK-COFF-DWO-DIR-DEFAULT: "/dwodir:a.out_dwo"
