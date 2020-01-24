// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -ffunction-sections -o - < %s | FileCheck %s --check-prefix=FUNC_SECT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -fdata-sections -o - < %s | FileCheck %s --check-prefix=DATA_SECT

// Try again through a clang invocation of the ThinLTO backend.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O2 %s -flto=thin -emit-llvm-bc -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -S -ffunction-sections -o - | FileCheck %s --check-prefix=FUNC_SECT
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -S -fdata-sections -o - | FileCheck %s --check-prefix=DATA_SECT

const int hello = 123;
void world() {}

// PLAIN-NOT: section
// PLAIN: world:
// PLAIN: section .rodata,
// PLAIN: hello:

// FUNC_SECT: section .text.world,
// FUNC_SECT: world:
// FUNC_SECT: section .rodata,
// FUNC_SECT: hello:

// DATA_SECT-NOT: .section
// DATA_SECT: world:
// DATA_SECT: .section .rodata.hello,
// DATA_SECT: hello:
