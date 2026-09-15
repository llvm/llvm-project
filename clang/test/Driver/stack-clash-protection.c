// RUN: %clang --target=i386-unknown-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang --target=i386-unknown-linux -fno-stack-clash-protection -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang --target=i386-unknown-linux -fstack-clash-protection -fno-stack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not='"-fstack-clash-protection"'

// RUN: %clang --target=x86_64-scei-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang --target=x86_64-unknown-freebsd -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED

// RUN: %clang --target=x86_64-unknown-linux -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not='"-fstack-clash-protection"'

// RUN: %clang --target=aarch64-linux-android -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// 32-bit Arm does not yet support stack clash protection; see
// https://github.com/llvm/llvm-project/issues/192533.
// RUN: %clang --target=armv7-linux-androideabi -### %s 2>&1 | FileCheck %s --check-prefix=DISABLED --implicit-check-not='"-fstack-clash-protection"'

// --implicit-check-not needs a positive directive to attach to. "-cc1" also
// proves a compilation job was emitted, so DISABLED cannot pass merely
// because clang printed nothing.
// ENABLED: "-fstack-clash-protection"
// DISABLED: "-cc1"

// RUN: %clang --target=armv7k-apple-linux -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=SCP-armv7 --implicit-check-not='"-fstack-clash-protection"'
// SCP-armv7: argument unused during compilation: '-fstack-clash-protection'

// RUN: %clang --target=x86_64-unknown-linux -fstack-clash-protection -S -emit-llvm -o %t.ll %s 2>&1 | FileCheck %s -check-prefix=SCP-warn
// SCP-warn: warning: unable to protect inline asm that clobbers stack pointer against stack clash

// RUN: %clang --target=x86_64-pc-unknown-linux -fstack-clash-protection -S -emit-llvm -o- %s | FileCheck %s -check-prefix=SCP-ll-linux64
// SCP-ll-linux64: attributes {{.*}} "probe-stack"="inline-asm"

// RUN: %clang --target=x86_64-pc-windows-msvc -fstack-clash-protection -S -emit-llvm -o- %s 2>&1 | FileCheck %s -check-prefix=SCP-ll-win64
// SCP-ll-win64-NOT: attributes {{.*}} "probe-stack"="inline-asm"
// SCP-ll-win64: argument unused during compilation: '-fstack-clash-protection'

// RUN: %clang --target=x86_64-unknown-fuchsia -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang --target=aarch64-unknown-fuchsia -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED
// RUN: %clang --target=riscv64-unknown-fuchsia -fstack-clash-protection -### %s 2>&1 | FileCheck %s --check-prefix=ENABLED

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
