// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/tls-opt-iele-i686-nopic.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %tso
// RUN: ld.lld -shared %t.o %tso -o %t1
// RUN: llvm-readobj -s -r -d %t1 | FileCheck --check-prefix=GOTRELSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASMSHARED %s

// GOTRELSHARED:     Section {
// GOTRELSHARED:      Index: 8
// GOTRELSHARED:      Name: .got
// GOTRELSHARED-NEXT:   Type: SHT_PROGBITS
// GOTRELSHARED-NEXT:   Flags [
// GOTRELSHARED-NEXT:     SHF_ALLOC
// GOTRELSHARED-NEXT:     SHF_WRITE
// GOTRELSHARED-NEXT:   ]
// GOTRELSHARED-NEXT:   Address: 0x1060
// GOTRELSHARED-NEXT:   Offset: 0x1060
// GOTRELSHARED-NEXT:   Size: 16
// GOTRELSHARED-NEXT:   Link: 0
// GOTRELSHARED-NEXT:   Info: 0
// GOTRELSHARED-NEXT:   AddressAlignment: 4
// GOTRELSHARED-NEXT:   EntrySize: 0
// GOTRELSHARED-NEXT: }
// GOTRELSHARED:      Relocations [
// GOTRELSHARED-NEXT:   Section ({{.*}}) .rel.dyn {
// GOTRELSHARED-NEXT:     0x2002 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x200A R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2013 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x201C R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2024 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x202D R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2036 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x203F R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x1060 R_386_TLS_TPOFF tlslocal0 0x0
// GOTRELSHARED-NEXT:     0x1064 R_386_TLS_TPOFF tlslocal1 0x0
// GOTRELSHARED-NEXT:     0x1068 R_386_TLS_TPOFF tlsshared0 0x0
// GOTRELSHARED-NEXT:     0x106C R_386_TLS_TPOFF tlsshared1 0x0
// GOTRELSHARED-NEXT:   }
// GOTRELSHARED-NEXT: ]
// GOTRELSHARED:      0x6FFFFFFA RELCOUNT             8

// DISASMSHARED:       Disassembly of section test:
// DISASMSHARED-NEXT:  _start:
// (.got)[0] = 0x1060 = 4192
// (.got)[1] = 0x1064 = 4196
// (.got)[2] = 0x1068 = 4200
// (.got)[3] = 0x106C = 4204
// DISASMSHARED-NEXT:  2000: {{.*}}  movl  4192, %ecx
// DISASMSHARED-NEXT:  2006: {{.*}}  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2009: {{.*}}  movl  4192, %eax
// DISASMSHARED-NEXT:  200e: {{.*}}  movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  2011: {{.*}}  addl  4192, %ecx
// DISASMSHARED-NEXT:  2017: {{.*}}  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  201a: {{.*}}  movl  4196, %ecx
// DISASMSHARED-NEXT:  2020: {{.*}}  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2023: {{.*}}  movl  4196, %eax
// DISASMSHARED-NEXT:  2028: {{.*}}  movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  202b: {{.*}}  addl  4196, %ecx
// DISASMSHARED-NEXT:  2031: {{.*}}  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2034: {{.*}}  movl  4200, %ecx
// DISASMSHARED-NEXT:  203a: {{.*}}  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  203d: {{.*}}  addl  4204, %ecx
// DISASMSHARED-NEXT:  2043: {{.*}}  movl  %gs:(%ecx), %eax

.type tlslocal0,@object
.section .tbss,"awT",@nobits
.globl tlslocal0
.align 4
tlslocal0:
 .long 0
 .size tlslocal0, 4

.type tlslocal1,@object
.section .tbss,"awT",@nobits
.globl tlslocal1
.align 4
tlslocal1:
 .long 0
 .size tlslocal1, 4

.section .text
.globl ___tls_get_addr
.type ___tls_get_addr,@function
___tls_get_addr:

.section test, "axw"
.globl _start
_start:
movl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal0@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal0@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlslocal1@indntpoff,%eax
movl %gs:(%eax),%eax

addl tlslocal1@indntpoff,%ecx
movl %gs:(%ecx),%eax

movl tlsshared0@indntpoff,%ecx
movl %gs:(%ecx),%eax

addl tlsshared1@indntpoff,%ecx
movl %gs:(%ecx),%eax
