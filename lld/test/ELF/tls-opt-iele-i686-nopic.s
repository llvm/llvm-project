// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/tls-opt-iele-i686-nopic.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %tso
// RUN: ld.lld %t.o %tso -o %t1
// RUN: llvm-readobj -s -r %t1 | FileCheck --check-prefix=GOTREL %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s
// RUN: ld.lld -shared %t.o %tso -o %t1
// RUN: llvm-readobj -s -r %t1 | FileCheck --check-prefix=GOTRELSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASMSHARED %s

// GOTREL:      Section {
// GOTREL:        Index:
// GOTREL:        Name: .got
// GOTREL-NEXT:   Type: SHT_PROGBITS
// GOTREL-NEXT:   Flags [
// GOTREL-NEXT:     SHF_ALLOC
// GOTREL-NEXT:     SHF_WRITE
// GOTREL-NEXT:   ]
// GOTREL-NEXT:   Address: 0x12050
// GOTREL-NEXT:   Offset: 0x2050
// GOTREL-NEXT:   Size: 8
// GOTREL-NEXT:   Link: 0
// GOTREL-NEXT:   Info: 0
// GOTREL-NEXT:   AddressAlignment: 4
// GOTREL-NEXT:   EntrySize: 0
// GOTREL-NEXT: }
// GOTREL:      Relocations [
// GOTREL-NEXT: Section ({{.*}}) .rel.dyn {
// GOTREL-NEXT:   0x12050 R_386_TLS_TPOFF tlsshared0 0x0
// GOTREL-NEXT:   0x12054 R_386_TLS_TPOFF tlsshared1 0x0
// GOTREL-NEXT:  }
// GOTREL-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-NEXT: _start:
// 4294967288 = 0xFFFFFFF8
// 4294967292 = 0xFFFFFFFC
// 73808 = (.got)[0] = 0x12050
// 73812 = (.got)[1] = 0x12054
// DISASM-NEXT: 11000: c7 c1 f8 ff ff ff movl $4294967288, %ecx
// DISASM-NEXT: 11006: 65 8b 01          movl %gs:(%ecx), %eax
// DISASM-NEXT: 11009: b8 f8 ff ff ff    movl $4294967288, %eax
// DISASM-NEXT: 1100e: 65 8b 00          movl %gs:(%eax), %eax
// DISASM-NEXT: 11011: 81 c1 f8 ff ff ff addl $4294967288, %ecx
// DISASM-NEXT: 11017: 65 8b 01          movl %gs:(%ecx), %eax
// DISASM-NEXT: 1101a: c7 c1 fc ff ff ff movl $4294967292, %ecx
// DISASM-NEXT: 11020: 65 8b 01          movl %gs:(%ecx), %eax
// DISASM-NEXT: 11023: b8 fc ff ff ff    movl $4294967292, %eax
// DISASM-NEXT: 11028: 65 8b 00          movl %gs:(%eax), %eax
// DISASM-NEXT: 1102b: 81 c1 fc ff ff ff addl $4294967292, %ecx
// DISASM-NEXT: 11031: 65 8b 01          movl %gs:(%ecx), %eax
// DISASM-NEXT: 11034: 8b 0d 50 20 01 00 movl 73808, %ecx
// DISASM-NEXT: 1103a: 65 8b 01          movl %gs:(%ecx), %eax
// DISASM-NEXT: 1103d: 03 0d 54 20 01 00 addl 73812, %ecx
// DISASM-NEXT: 11043: 65 8b 01          movl %gs:(%ecx), %eax

// GOTRELSHARED:     Section {
// GOTRELSHARED:      Index: 8
// GOTRELSHARED:      Name: .got
// GOTRELSHARED-NEXT:   Type: SHT_PROGBITS
// GOTRELSHARED-NEXT:   Flags [
// GOTRELSHARED-NEXT:     SHF_ALLOC
// GOTRELSHARED-NEXT:     SHF_WRITE
// GOTRELSHARED-NEXT:   ]
// GOTRELSHARED-NEXT:   Address: 0x2050
// GOTRELSHARED-NEXT:   Offset: 0x2050
// GOTRELSHARED-NEXT:   Size: 16
// GOTRELSHARED-NEXT:   Link: 0
// GOTRELSHARED-NEXT:   Info: 0
// GOTRELSHARED-NEXT:   AddressAlignment: 4
// GOTRELSHARED-NEXT:   EntrySize: 0
// GOTRELSHARED-NEXT: }
// GOTRELSHARED:      Relocations [
// GOTRELSHARED-NEXT:   Section ({{.*}}) .rel.dyn {
// GOTRELSHARED-NEXT:     0x1002 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2050 R_386_TLS_TPOFF tlslocal0 0x0
// GOTRELSHARED-NEXT:     0x100A R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x1013 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x101C R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2054 R_386_TLS_TPOFF tlslocal1 0x0
// GOTRELSHARED-NEXT:     0x1024 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x102D R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x1036 R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x2058 R_386_TLS_TPOFF tlsshared0 0x0
// GOTRELSHARED-NEXT:     0x103F R_386_RELATIVE - 0x0
// GOTRELSHARED-NEXT:     0x205C R_386_TLS_TPOFF tlsshared1 0x0
// GOTRELSHARED-NEXT:   }
// GOTRELSHARED-NEXT: ]

// DISASMSHARED:       Disassembly of section .text:
// DISASMSHARED-NEXT:  _start:
// (.got)[0] = 0x2050 = 8272
// (.got)[1] = 0x2054 = 8276
// (.got)[2] = 0x2058 = 8280
// (.got)[3] = 0x205C = 8284
// DISASMSHARED-NEXT:  1000: 8b 0d 50 20 00 00 movl 8272, %ecx
// DISASMSHARED-NEXT:  1006: 65 8b 01          movl %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  1009: a1 50 20 00 00    movl 8272, %eax
// DISASMSHARED-NEXT:  100e: 65 8b 00          movl %gs:(%eax), %eax
// DISASMSHARED-NEXT:  1011: 03 0d 50 20 00 00 addl 8272, %ecx
// DISASMSHARED-NEXT:  1017: 65 8b 01          movl %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  101a: 8b 0d 54 20 00 00 movl 8276, %ecx
// DISASMSHARED-NEXT:  1020: 65 8b 01          movl %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  1023: a1 54 20 00 00    movl 8276, %eax
// DISASMSHARED-NEXT:  1028: 65 8b 00          movl %gs:(%eax), %eax
// DISASMSHARED-NEXT:  102b: 03 0d 54 20 00 00 addl 8276, %ecx
// DISASMSHARED-NEXT:  1031: 65 8b 01          movl %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  1034: 8b 0d 58 20 00 00 movl 8280, %ecx
// DISASMSHARED-NEXT:  103a: 65 8b 01          movl %gs:(%ecx), %eax
// DISASMSHARED-NEXT:  103d: 03 0d 5c 20 00 00 addl 8284, %ecx
// DISASMSHARED-NEXT:  1043: 65 8b 01          movl %gs:(%ecx), %eax

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

.section .text
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
