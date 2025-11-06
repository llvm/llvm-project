// RUN: %clang --target=i386-unknown-linux -masm=intel -c %s -### 2>&1 | FileCheck --check-prefix=CHECK-INTEL %s
// RUN: %clang --target=i386-unknown-linux -masm=att -c %s -### 2>&1 | FileCheck --check-prefix=CHECK-ATT %s
// RUN: not %clang --target=i386-unknown-linux -c -masm=somerequired %s -### 2>&1 | FileCheck --check-prefix=CHECK-SOMEREQUIRED %s
// RUN: %clang --target=arm-unknown-eabi -c -masm=intel %s -### 2>&1 | FileCheck --check-prefix=CHECK-ARM %s

// CHECK-INTEL: -x86-asm-syntax=intel
// CHECK-ATT: -x86-asm-syntax=att
// CHECK-SOMEREQUIRED: error: unsupported argument 'somerequired' to option '-masm='
// CHECK-ARM: warning: argument unused during compilation: '-masm=intel'
.text
mov    al, 0
