// RUN: %clang -target i386-unknown-linux -masm=intel -S %s -### 2>&1 | FileCheck --check-prefix=CHECK-INTEL %s
// RUN: %clang -target i386-unknown-linux -masm=att -S %s -### 2>&1 | FileCheck --check-prefix=CHECK-ATT %s
// RUN: %clang -target i386-unknown-linux -S -masm=somerequired %s -### 2>&1 | FileCheck --check-prefix=CHECK-SOMEREQUIRED %s
// RUN: %clang -target arm-unknown-eabi -S -masm=intel %s -### 2>&1 | FileCheck --check-prefix=CHECK-ARM %s
// RUN: %clang_cl --target=x86_64 /FA -### -- %s 2>&1 | FileCheck --check-prefix=CHECK-CL %s

int f() {
// CHECK-INTEL: -x86-asm-syntax=intel
// CHECK-ATT: -x86-asm-syntax=att
// CHECK-SOMEREQUIRED: error: unsupported argument 'somerequired' to option 'masm='
// CHECK-ARM: warning: argument unused during compilation: '-masm=intel'
// CHECK-CL: -x86-asm-syntax=intel
  return 0;
}
