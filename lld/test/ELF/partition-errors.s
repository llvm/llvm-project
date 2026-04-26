// REQUIRES: x86, mips
/// Link against a DSO to ensure that sections are not discarded by --gc-sections.
// RUN: llvm-mc %S/Inputs/shared.s -o %ts.o -filetype=obj --triple=x86_64
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc -triple=x86_64-unknown-linux -filetype=obj -o %t.o %s
// RUN: echo "SECTIONS {}" > %t.script
// RUN: not ld.lld --export-dynamic %ts.so %t.o %t.script 2>&1 | FileCheck %s
// RUN: echo "PHDRS { text PT_LOAD; }" > %t2.script
// RUN: not ld.lld --export-dynamic %ts.so %t.o %t2.script 2>&1 | FileCheck %s
// RUN: not ld.lld --export-dynamic %ts.so %t.o --section-start .text=0 2>&1 | FileCheck %s
// RUN: not ld.lld --export-dynamic %ts.so %t.o -Ttext=0 2>&1 | FileCheck %s
// RUN: not ld.lld --export-dynamic %ts.so %t.o -Tdata=0 2>&1 | FileCheck %s
// RUN: not ld.lld --export-dynamic %ts.so %t.o -Tbss=0 2>&1 | FileCheck %s

// RUN: llvm-mc %S/Inputs/shared.s -o %ts.o -filetype=obj --triple=mipsel-unknown-linux
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc -triple=mipsel-unknown-linux -filetype=obj -o %t2.o %s
// RUN: not ld.lld --export-dynamic %t2.o %ts.so 2>&1 | FileCheck %s

// CHECK: error: {{.*}}.o: partitions cannot be used

.section .llvm_sympart.f1,"",@llvm_sympart
.asciz "part1"
.quad f1

.text
.globl f1
f1:
