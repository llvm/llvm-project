// REQUIRES: x86
/// Link against a DSO to ensure that sections are not discarded by --gc-sections.
// RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o %ts.o
// RUN: ld.lld -shared -soname=ts %ts.o -o %ts.so
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o %ts.so -o %t --export-dynamic --gc-sections
// RUN: llvm-readelf -S %t | FileCheck --implicit-check-not=has_startstop --implicit-check-not=no_startstop %s

/// All input sections now live in the main partition, so a section that
/// could otherwise have been split (no_startstop) is also merged into a
/// single output section. __start_/__stop_ semantics keep working naturally.

// CHECK: has_startstop
// CHECK: no_startstop

.section .llvm_sympart.f1,"",@llvm_sympart
.asciz "part1"
.quad f1

.section .text._start,"ax",@progbits
.globl _start
_start:
call __start_has_startstop
call __stop_has_startstop

.section .text.f1,"ax",@progbits
.globl f1
f1:

.section has_startstop,"ao",@progbits,.text._start,unique,1
.quad 1

.section has_startstop,"ao",@progbits,.text.f1,unique,2
.quad 2

.section no_startstop,"ao",@progbits,.text._start,unique,1
.quad 3

.section no_startstop,"ao",@progbits,.text.f1,unique,2
.quad 4
