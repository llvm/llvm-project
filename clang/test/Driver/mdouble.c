// RUN: %clang -target avr -c -### %s -mdouble=64 2>&1 | FileCheck %s

// CHECK: "-mdouble=64"

// RUN: not %clang --target=aarch64 -c -### %s -mdouble=64 2>&1 | FileCheck --check-prefix=ERR %s

// ERR: error: unsupported option '-mdouble=64' for target 'aarch64'
