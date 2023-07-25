# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.la64.o

# RUN: ld.lld --section-start=.rodata=0x1234567890 --section-start=.text=0x9876543210 %t.la64.o -o %t.la64
# RUN: llvm-readelf -x .rodata %t.la64 | FileCheck --check-prefix=CHECK %s
# CHECK:      section '.rodata':
# CHECK-NEXT: 0x1234567890 10325476 98badcfe 80b9fd41 86000000
# CHECK-NEXT: 0x12345678a0 80b9fd41 80b980

.global _start

_start:
1:
    break 0

.rodata
2:
    .dword 0xfedcba9876543210

foo:
    .dword 0
    .reloc foo, R_LARCH_ADD64, 1b
    .reloc foo, R_LARCH_SUB64, 2b
bar:
    .word 0
    .reloc bar, R_LARCH_ADD32, 1b
    .reloc bar, R_LARCH_SUB32, 2b
baz:
    .short 0
    .reloc baz, R_LARCH_ADD16, 1b
    .reloc baz, R_LARCH_SUB16, 2b
quux:
    .byte 0
    .reloc quux, R_LARCH_ADD8, 1b
    .reloc quux, R_LARCH_SUB8, 2b
