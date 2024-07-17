// RUN: %clang --target=powerpc-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64-pc-freebsd12 -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=powerpc64le-linux-musl -c -### %s -mlong-double-128 2>&1 | FileCheck %s
// RUN: %clang --target=i686-linux-gnu -c -### %s -mlong-double-128 2>&1 | FileCheck %s

// RUN: %clang --target=x86_64-linux-musl -c -### %s -mlong-double-128 -mlong-double-80 2>&1 | FileCheck --implicit-check-not=-mlong-double-128 /dev/null
// RUN: %clang --target=x86_64-linux-musl -c -### %s -mlong-double-80 -mlong-double-128 2>&1 | FileCheck %s

// CHECK: "-mlong-double-128"

// RUN: not %clang --target=aarch64 -c -### %s -mlong-double-128 2>&1 | FileCheck --check-prefix=ERR %s
// RUN: not %clang --target=powerpc -c -### %s -mlong-double-80 2>&1 | FileCheck --check-prefix=ERR2 %s

// ERR: error: unsupported option '-mlong-double-128' for target 'aarch64'
// ERR2: error: unsupported option '-mlong-double-80' for target 'powerpc'
