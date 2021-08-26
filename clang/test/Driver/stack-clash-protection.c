// RUN: %clang -target i386-unknown-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-i386
// RUN: %clang -target i386-unknown-linux -fno-stack-clash-protection -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-i386
// RUN: %clang -target i386-unknown-linux -fstack-clash-protection -fno-stack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-i386-NO
// SCP-i386: "-fstack-clash-protection"
// SCP-i386-NO-NOT: "-fstack-clash-protection"

// RUN: %clang -target x86_64-scei-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-x86
// RUN: %clang -target x86_64-unknown-freebsd -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-x86
// SCP-x86: "-fstack-clash-protection"

// RUN: %clang -target armv7k-apple-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SCP-armv7
// SCP-armv7-NOT: "-fstack-clash-protection"
// SCP-armv7: argument unused during compilation: '-fstack-clash-protection'

// RUN: %clang -target x86_64-unknown-linux -fstack-clash-protection -S -emit-llvm -o %t.ll %s 2>&1 | FileCheck %s -check-prefix=SCP-warn
// SCP-warn: warning: Unable to protect inline asm that clobbers stack pointer against stack clash

// RUN: %clang -target x86_64-pc-unknown-linux -fstack-clash-protection -S -emit-llvm -o- %s | FileCheck %s -check-prefix=SCP-ll-linux64
// SCP-ll-linux64: attributes {{.*}} "probe-stack"="inline-asm"

// RUN: %clang -target x86_64-pc-windows-msvc -fstack-clash-protection -S -emit-llvm -o- %s 2>&1 | FileCheck %s -check-prefix=SCP-ll-win64
// SCP-ll-win64-NOT: attributes {{.*}} "probe-stack"="inline-asm"
// SCP-ll-win64: argument unused during compilation: '-fstack-clash-protection'

int foo(int c) {
  int r;
  __asm__("sub %0, %%rsp"
          :
          : "rm"(c)
          : "rsp");
  __asm__("mov %%rsp, %0"
          : "=rm"(r)::);
  return r;
}
