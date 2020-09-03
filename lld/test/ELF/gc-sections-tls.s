# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## Relocation in a non .debug_* referencing a discarded TLS symbol is invalid.
## If we happen to have no PT_TLS, we will emit an error.
# RUN: not ld.lld %t.o --gc-sections -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: {{.*}}.o has an STT_TLS symbol but doesn't have an SHF_TLS section

## If we happen to have a PT_TLS, we will resolve the relocation to
## an arbitrary value (current implementation uses a negative value).
# RUN: echo '.section .tbss,"awT"; .globl root; root: .long 0' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld --gc-sections -u root %t.o %t1.o -o %t
# RUN: llvm-readelf -x .noalloc %t | FileCheck %s

# CHECK:      Hex dump of section '.noalloc':
# CHECK-NEXT: 0x00000000 {{[0-9a-f]+}} ffffffff

.section .tbss,"awT",@nobits
tls:
  .long 0

.section .noalloc,""
  .quad tls+0x8000
