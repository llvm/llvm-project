// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/tls-opt-iele-i686-nopic.s -o %tso.o
// RUN: ld.lld -shared -soname=t.so %tso.o -o %tso
// RUN: ld.lld -shared %t.o %tso -o %t1
// RUN: llvm-readobj -S -r -d %t1 | FileCheck --check-prefix=GOTRELSHARED %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t1 | FileCheck --check-prefix=DISASMSHARED %s

// GOTRELSHARED:     Section {
// GOTRELSHARED:      Name: .got
// GOTRELSHARED-NEXT:   Type: SHT_PROGBITS
// GOTRELSHARED-NEXT:   Flags [
// GOTRELSHARED-NEXT:     SHF_ALLOC
// GOTRELSHARED-NEXT:     SHF_WRITE
// GOTRELSHARED-NEXT:   ]
// GOTRELSHARED-NEXT:   Address: 0x3388
// GOTRELSHARED-NEXT:   Offset: 0x388
// GOTRELSHARED-NEXT:   Size: 16
// GOTRELSHARED-NEXT:   Link: 0
// GOTRELSHARED-NEXT:   Info: 0
// GOTRELSHARED-NEXT:   AddressAlignment: 4
// GOTRELSHARED-NEXT:   EntrySize: 0
// GOTRELSHARED-NEXT: }
// GOTRELSHARED:      0x6FFFFFFA RELCOUNT             8
// GOTRELSHARED:      Relocations [
// GOTRELSHARED-NEXT:   Section ({{.*}}) .rel.dyn {
// GOTRELSHARED-NEXT:     0x22DA R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x22E2 R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x22EB R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x22F4 R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x22FC R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x2305 R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x230E R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x2317 R_386_RELATIVE -
// GOTRELSHARED-NEXT:     0x3390 R_386_TLS_TPOFF tlsshared0
// GOTRELSHARED-NEXT:     0x3394 R_386_TLS_TPOFF tlsshared1
// GOTRELSHARED-NEXT:     0x3388 R_386_TLS_TPOFF tlslocal0
// GOTRELSHARED-NEXT:     0x338C R_386_TLS_TPOFF tlslocal1
// GOTRELSHARED-NEXT:   }
// GOTRELSHARED-NEXT: ]

// DISASMSHARED:       Disassembly of section test:
// DISASMSHARED-EMPTY:
// DISASMSHARED-NEXT:  <_start>:
// (.got)[0] = 0x3388 = 13192
// (.got)[1] = 13196
// (.got)[2] = 13200
// (.got)[3] = 13204
// DISASMSHARED-NEXT:  22d8:       movl  13192, %ecx
// DISASMSHARED-NEXT:  22de:       movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  22e1:       movl  13192, %eax
// DISASMSHARED-NEXT:  22e6:       movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  22e9:       addl  13192, %ecx
// DISASMSHARED-NEXT:  22ef:       movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  22f2:       movl  13196, %ecx
// DISASMSHARED-NEXT:  22f8:       movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  22fb:       movl  13196, %eax
// DISASMSHARED-NEXT:  2300:       movl  %gs:(%eax), %eax
// DISASMSHARED-NEXT:  2303:       addl  13196, %ecx
// DISASMSHARED-NEXT:  2309:       movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  230c:       movl  13200, %ecx
// DISASMSHARED-NEXT:  2312:       movl  %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  2315:       addl  13204, %ecx
// DISASMSHARED-NEXT:  231b:       movl  %gs:(%ecx), %eax

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
