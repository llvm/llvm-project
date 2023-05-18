// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o -o %t --export-dynamic --gc-sections
// RUN: llvm-readelf -S %t | FileCheck --implicit-check-not=has_startstop %s

// We can't let the has_startstop section be split by partition because it is
// referenced by __start_ and __stop_ symbols, so the split could result in
// some sections being moved out of the __start_/__stop_ range. Make sure that
// that didn't happen by checking that there is only one section.
//
// It's fine for us to split no_startstop because of the lack of
// __start_/__stop_ symbols.

// CHECK: has_startstop
// CHECK: no_startstop

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
