// This test check that orbis-ld is used for linker all the time. Specifying
// linker using -fuse-ld causes a error message emitted and compilation fail.

// REQUIRES: system-windows, x86-registered-target

// RUN: mkdir -p %t
// RUN: touch %t/orbis-ld.exe

// RUN: env "PATH=%t;%PATH%;" %clang -target x86_64-scei-ps4  %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%t;%PATH%;" %clang -target x86_64-scei-ps4  %s -shared -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s

// CHECK-PS4-LINKER: \\orbis-ld

// RUN: env "PATH=%t;%PATH%;" %clang -### -target x86_64-scei-ps4 -flto=thin %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LTO-THIN %s
// RUN: env "PATH=%t;%PATH%;" %clang -### -target x86_64-scei-ps4 -flto=full %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LTO-FULL %s
// RUN: env "PATH=%t;%PATH%;" %clang -### -target x86_64-scei-ps4  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-NO-LTO-THIN --check-prefix=CHECK-PS4-NO-LTO-FULL %s

// CHECK-PS4-LTO-THIN: --lto=thin
// CHECK-PS4-LTO-FULL: --lto=full
// CHECK-PS4-NO-LTO-THIN-NOT: --lto=thin
// CHECK-PS4-NO-LTO-FULL-NOT: --lto=full

// RUN: env "PATH=%t;%PATH%;" %clang -target x86_64-scei-ps4 %s -fuse-ld=gold -### 2>&1 \
// RUN:   | FileCheck --check-prefix=ERROR %s

// ERROR: error: unsupported option '-fuse-ld' for target 'x86_64-scei-ps4'
