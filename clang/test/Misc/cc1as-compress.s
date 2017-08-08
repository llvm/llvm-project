// REQUIRES: zlib
// REQUIRES: x86-registered-target

// RUN: %clang -cc1as -triple i686 --compress-debug-sections -filetype asm %s -o /dev/null 2>&1 | FileCheck -allow-empty %s
// RUN: %clang -cc1as -triple i686 -compress-debug-sections -filetype asm %s -o /dev/null 2>&1 | FileCheck -allow-empty %s

// CHECK-NOT: error: unknown argument:

