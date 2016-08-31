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
// GOTRELSHARED-NEXT:   Address: 0x1058
// GOTRELSHARED-NEXT:   Offset: 0x1058
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
// GOTRELSHARED-NEXT:     0x1058 R_386_TLS_TPOFF tlslocal0 0x0
// GOTRELSHARED-NEXT:     0x105C R_386_TLS_TPOFF tlslocal1 0x0
// GOTRELSHARED-NEXT:     0x1060 R_386_TLS_TPOFF tlsshared0 0x0
// GOTRELSHARED-NEXT:     0x1064 R_386_TLS_TPOFF tlsshared1 0x0
// GOTRELSHARED-NEXT:   }
// GOTRELSHARED-NEXT: ]
// GOTRELSHARED:      0x6FFFFFFA RELCOUNT             8

// DISASMSHARED:       Disassembly of section test:
// DISASMSHARED-NEXT:  _start:
// (.got)[0] = 0x2050 = 8272
// (.got)[1] = 0x2054 = 8276
// (.got)[2] = 0x2058 = 8280
// (.got)[3] = 0x205C = 8284
// DISASMSHARED-NEXT:  2000: 8b 0d 58 10 00 00   movl  4184, %ecx
// DISASMSHARED-NEXT:  2006: 65 8b 01  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2009: a1 58 10 00 00  movl  4184, %eax
// DISASMSHARED-NEXT:  200e: 65 8b 00  movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  2011: 03 0d 58 10 00 00   addl  4184, %ecx
// DISASMSHARED-NEXT:  2017: 65 8b 01  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  201a: 8b 0d 5c 10 00 00   movl  4188, %ecx
// DISASMSHARED-NEXT:  2020: 65 8b 01  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2023: a1 5c 10 00 00  movl  4188, %eax
// DISASMSHARED-NEXT:  2028: 65 8b 00  movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  202b: 03 0d 5c 10 00 00   addl  4188, %ecx
// DISASMSHARED-NEXT:  2031: 65 8b 01  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2034: 8b 0d 60 10 00 00   movl  4192, %ecx
// DISASMSHARED-NEXT:  203a: 65 8b 01  movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  203d: 03 0d 64 10 00 00   addl  4196, %ecx
// DISASMSHARED-NEXT:  2043: 65 8b 01  movl  %gs:(%ecx), %eax

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
