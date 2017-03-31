// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -ffunction-sections -fno-function-sections -o - < %s | FileCheck %s --check-prefix=PLAIN

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -ffunction-sections -o - < %s | FileCheck %s --check-prefix=FUNC_SECT
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -fno-function-sections -ffunction-sections -o - < %s | FileCheck %s --check-prefix=FUNC_SECT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -fdata-sections -o - < %s | FileCheck %s --check-prefix=DATA_SECT
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -fno-data-sections -fdata-sections -o - < %s | FileCheck %s --check-prefix=DATA_SECT

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
