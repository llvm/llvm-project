# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj /out:%t.exe /entry:main /subsystem:console
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# CHECK: Contents of section .text:
.globl main
main:
# CHECK-NEXT: 140001000 00200040 01000000 01200040 01000000
.8byte _ZTS
.8byte _ZTV
# CHECK-NEXT: 140001010 01200040 01000000
.8byte _ZTVa

.section .rdata,"dr",discard,_ZTS
.globl _ZTS
_ZTS:
.byte 42

.section .rdata,"dr",discard,_ZTV
.globl _ZTV
_ZTV:
.byte 42

.section .rdata,"dr",discard,_ZTVa
.globl _ZTVa
_ZTVa:
.byte 42
