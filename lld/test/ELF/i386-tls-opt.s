// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
// RUN: ld.lld %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASM %s

// NORELOC:      Relocations [
// NORELOC-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// LD -> LE:
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   nop
// DISASM-NEXT:   leal (%esi,%eiz), %esi
// DISASM-NEXT:   leal -8(%eax), %edx
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   nop
// DISASM-NEXT:   leal (%esi,%eiz), %esi
// DISASM-NEXT:   leal -4(%eax), %edx
// IE -> LE:
// 4294967288 == 0xFFFFFFF8
// 4294967292 == 0xFFFFFFFC
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   movl $4294967288, %eax
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   movl $4294967292, %eax
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   leal -8(%eax), %eax
// DISASM-NEXT:   movl %gs:0, %eax
// DISASM-NEXT:   leal -4(%eax), %eax
.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
.align 4
tls0:
 .long 0
 .size tls0, 4

.type  tls1,@object
.globl tls1
.align 4
tls1:
 .long 0
 .size tls1, 4

.section .text
.globl ___tls_get_addr
.type ___tls_get_addr,@function
___tls_get_addr:

.section .text
.globl _start
_start:
//LD -> LE:
leal tls0@tlsldm(%ebx),%eax
call ___tls_get_addr@plt
leal tls0@dtpoff(%eax),%edx
leal tls1@tlsldm(%ebx),%eax
call ___tls_get_addr@plt
leal tls1@dtpoff(%eax),%edx
//IE -> LE:
movl %gs:0,%eax
movl tls0@gotntpoff(%ebx),%eax
movl %gs:0,%eax
movl tls1@gotntpoff(%ebx),%eax
movl %gs:0,%eax
addl tls0@gotntpoff(%ebx),%eax
movl %gs:0,%eax
addl tls1@gotntpoff(%ebx),%eax
