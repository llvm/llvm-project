// RUN: %clang -target powerpc-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang -target powerpc64-pc-freebsd12 -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang -target powerpc64le-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang -target i686-linux-gnu -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang -target x86_64-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s

// CHECK: "-mlong-double-128"

// RUN: %clang -target aarch64 -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR %s

// ERR: error: unsupported option '-mlong-double-128' for target 'aarch64'
