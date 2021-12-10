// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fasm-blocks -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -fasm-blocks -emit-llvm %s -o - | FileCheck %s

// This test is designed to check if we use the mem size info for parsing MS
// InlineAsm which use a global variable and one/two registers in a memory
// expression. If we not use this mem size info, there will be error of
// ambiguous operand size for some instructions. (e.g. 'fmul')
__attribute__((aligned (16)))
static const unsigned int static_const_table[] = { 0x00800000, };


void __attribute__ ((naked)) foo(void)
{__asm{
    fmul qword ptr [static_const_table + 0x00f0 +edx]
    ret
}}

// CHECK-LABEL: foo
// CHECK: call void asm sideeffect inteldialect "fmul qword ptr static_const_table[edx + $$240]\0A\09ret"
