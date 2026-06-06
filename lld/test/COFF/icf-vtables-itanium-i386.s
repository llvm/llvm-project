# REQUIRES: x86
# RUN: llvm-mc -triple=i386-windows-gnu -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj /out:%t.exe /entry:main /subsystem:console /safeseh:no
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# CHECK: Contents of section .text:
.globl _main
_main:
# CHECK-NEXT: 401000 00204000 01204000 01204000
.long __ZTS
.long __ZTV
.long __ZTVa

.section .rdata,"dr",discard,__ZTS
.globl __ZTS
__ZTS:
.byte 42

.section .rdata,"dr",discard,__ZTV
.globl __ZTV
__ZTV:
.byte 42

.section .rdata,"dr",discard,__ZTVa
.globl __ZTVa
__ZTVa:
.byte 42
