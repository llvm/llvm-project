# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## When a TLS section is discarded, we will resolve the relocation in a non-SHF_ALLOC
## section to the addend. Technically, we can emit an error in this case as the
## relocation type is not TLS.
# RUN: ld.lld %t.o --gc-sections -o %t
# RUN: llvm-readelf -x .noalloc %t | FileCheck %s

# RUN: echo '.section .tbss,"awT"; .globl root; root: .long 0' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld --gc-sections -u root %t.o %t1.o -o %t
# RUN: llvm-readelf -x .noalloc %t | FileCheck %s

# CHECK:      Hex dump of section '.noalloc':
# CHECK-NEXT: 0x00000000 00800000 00000000

.globl _start
_start:

.section .tbss,"awT",@nobits
  .long 0
tls:
  .long 0

.section .noalloc,""
  .quad tls+0x8000
