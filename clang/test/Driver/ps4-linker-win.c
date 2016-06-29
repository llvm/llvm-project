// The full path to the gold linker was not found on Windows because the
// driver fails to add an .exe extension to the name.
// We check that gold linker's full name (with an extension) is specified
// on the command line if -fuse-ld=gold, or -shared with no -fuse-ld option
// are passed. Otherwise, we check that the PS4's linker's full name is
// specified.

// REQUIRES: system-windows, x86-registered-target

// RUN: touch %T/orbis-ld.exe
// RUN: touch %T/orbis-ld.gold.exe

// RUN: env "PATH=%T;%PATH%;" %clang -target x86_64-scei-ps4  %s -fuse-ld=gold -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-GOLD %s
// RUN: env "PATH=%T;%PATH%;" %clang -target x86_64-scei-ps4  %s -shared -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-GOLD %s

// RUN: env "PATH=%T;%PATH%;" %clang -target x86_64-scei-ps4  %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%T;%PATH%;" %clang -target x86_64-scei-ps4  %s -fuse-ld=ps4 -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%T;%PATH%;" %clang -target x86_64-scei-ps4  %s -shared \
// RUN:     -fuse-ld=ps4 -### 2>&1 | FileCheck --check-prefix=CHECK-PS4-LINKER %s

// CHECK-PS4-GOLD: \\orbis-ld.gold
// CHECK-PS4-LINKER: \\orbis-ld
