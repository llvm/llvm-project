/// Checks proper linker prefixing for PS4 and PS5.
// UNSUPPORTED: system-windows

// RUN: mkdir -p %t
// RUN: rm -f %t/orbis-ld
// RUN: touch %t/orbis-ld
// RUN: chmod +x %t/orbis-ld
// RUN: rm -f %t/prospero-lld
// RUN: touch %t/prospero-lld
// RUN: chmod +x %t/prospero-lld

// RUN: env "PATH=%t:%PATH%" %clang -### -target x86_64-scei-ps4  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS4-LINKER %s
// RUN: env "PATH=%t:%PATH%" %clang -### -target x86_64-scei-ps4  %s -shared 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-PS4-LINKER,SHARED %s
// RUN: env "PATH=%t:%PATH%" %clang -### -target x86_64-sie-ps5  %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PS5-LINKER %s
// RUN: env "PATH=%t:%PATH%" %clang -### -target x86_64-sie-ps5  %s -shared 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK-PS5-LINKER,SHARED %s

// CHECK-PS4-LINKER: /orbis-ld
// CHECK-PS5-LINKER: /prospero-lld
// SHARED: "--shared"

// RUN: env "PATH=%t:%PATH%" not %clang -### --target=x86_64-scei-ps4 %s -fuse-ld=gold 2>&1 \
// RUN:   | FileCheck --check-prefix=ERROR %s
// RUN: env "PATH=%t:%PATH%" not %clang -### --target=x86_64-sie-ps5 %s -fuse-ld=gold 2>&1 \
// RUN:   | FileCheck --check-prefix=ERROR %s

// ERROR: error: unsupported option '-fuse-ld' for target 'x86_64-{{(scei|sie)}}-ps{{[45]}}'
