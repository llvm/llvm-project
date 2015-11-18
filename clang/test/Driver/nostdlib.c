// RUN: %clang -target i686-pc-linux-gnu -### -nostdlib %s 2> %t
// RUN: FileCheck < %t %s
//
// CHECK-NOT: start-group

// Most of the toolchains would check for -nostartfiles and -nostdlib
// in a short-circuiting boolean expression, so if both of the preceding
// options were present, the second would warn about being unused.
// RUN: %clang -### -nostartfiles -nostdlib -target i386-apple-darwin %s \
// RUN:   2>&1 | FileCheck %s -check-prefix=ARGSCLAIMED
// ARGSCLAIMED-NOT: warning:
