// RUN: %clang -### -target i686-pc-windows-gnu --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_LD_32 %s
// CHECK_LD_32: {{ld|ld.exe}}"
// CHECK_LD_32: "i386pe"
// CHECK_LD_32_NOT: "-flavor" "gnu"

// RUN: %clang -### -target i686-pc-windows-gnu --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -fuse-ld=lld 2>&1 | FileCheck -check-prefix=CHECK_LLD_32 %s
// CHECK_LLD_32: lld" "-flavor" "gnu"
// CHECK_LLD_32: "i386pe"

// RUN: %clang -### -target i686-pc-windows-gnu --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -fuse-ld=link.exe 2>&1 | FileCheck -check-prefix=CHECK_LINK_32 %s
// CHECK_LINK_32: link.exe"
// CHECK_LINK_32: "i386pe"

// RUN: %clang -### -target x86_64-pc-windows-gnu --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -fuse-ld=lld 2>&1 | FileCheck -check-prefix=CHECK_LLD_64 %s
// CHECK_LLD_64: lld" "-flavor" "gnu"
// CHECK_LLD_64: "i386pep"

// RUN: %clang -### -target arm-pc-windows-gnu --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -fuse-ld=lld 2>&1 | FileCheck -check-prefix=CHECK_LLD_ARM %s
// CHECK_LLD_ARM: lld" "-flavor" "gnu"
// CHECK_LLD_ARM: "thumb2pe"
