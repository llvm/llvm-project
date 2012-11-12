// RUN: %clang -target i686-pc-win32 -lkernel32.lib -luser32.lib -### %s 2>&1 | FileCheck %s
// CHECK-NOT: "-lkernel32.lib"
// CHECK-NOT: "-luser32.lib"
// CHECK: "kernel32.lib"
// CHECK: "user32.lib"
