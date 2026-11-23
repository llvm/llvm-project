// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-pc-linux -S -ffunction-sections -fdata-sections -fno-unique-section-names  -o - < %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-pc-linux -S -ffunction-sections -fdata-sections -fno-unique-section-names  -o - < %s | FileCheck %s %}
// RUN: %clang_cc1 -triple x86_64-pc-linux -S -ffunction-sections -fdata-sections -o - < %s | FileCheck %s --check-prefix=UNIQUE
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-pc-linux -S -ffunction-sections -fdata-sections -o - < %s | FileCheck %s --check-prefix=UNIQUE %}

const int hello = 123;
void world(void) {}

// CHECK: .section .text,"ax",@progbits,unique
// CHECK: .section .rodata,"a",@progbits,unique

// UNIQUE: .section .text.world,"ax",@progbits
// UNIQUE: .section .rodata.hello,"a",@progbits
