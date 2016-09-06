// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target

// RUN: mkdir -p %T/Output
// RUN: rm -f %T/Output/orbis-ld
// RUN: touch %T/Output/orbis-ld
// RUN: chmod +x %T/Output/orbis-ld

// RUN: env "PATH=%T/Output:%PATH%" %clang -### -target x86_64-scei-ps4  %s -fuse-ld=gold 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%T/Output:%PATH%" %clang -### -target x86_64-scei-ps4  %s -shared 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s

// RUN: env "PATH=%T/Output:%PATH%" %clang -### -target x86_64-scei-ps4  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%T/Output:%PATH%" %clang -### -target x86_64-scei-ps4  %s -fuse-ld=ps4 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%T/Output:%PATH%" %clang -### -target x86_64-scei-ps4  %s -shared \
// RUN:     -fuse-ld=ps4 2>&1 | FileCheck --check-prefix=CHECK-PS4-LINKER %s

// CHECK-PS4-LINKER: /orbis-ld
