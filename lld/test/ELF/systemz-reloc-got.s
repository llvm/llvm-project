# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld -z norelro -shared %t.o -soname=t.so -o %t.so
## Note: Without norelro the distance between .got and .got.plt causes
## R_390_GOTPLT12 relocations to always overflow.

# RUN: llvm-readelf -S -x .data %t.so | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=DISASM

# CHECK: Section Headers:
# CHECK: .got PROGBITS 0000000000002458
# CHECK: .got.plt PROGBITS 0000000000002480

## Note: _GLOBAL_OFFSET_TABLE is at .got
## GOT (foo) is at .got + 24 == 0x2470
## GOT (bar) is at .got + 32 == 0x2478
## GOTPLT (foo) is at .got.plt + 0 == .got + 40 == 0x2480
## GOTPLT (bar) is at .got.plt + 8 == .got + 48 == 0x2488

# DISASM:      larl %r12, 0x2458
# DISASM-NEXT: larl %r1, 0x2470
# DISASM-NEXT: larl %r1, 0x2478
# DISASM-NEXT: larl %r1, 0x2480
# DISASM-NEXT: larl %r1, 0x2488

# DISASM-NEXT: l %r1, 24(%r12)
# DISASM-NEXT: l %r1, 32(%r12)
# DISASM-NEXT: l %r1, 40(%r12)
# DISASM-NEXT: l %r1, 48(%r12)
# DISASM-NEXT: lg %r1, 24(%r12)
# DISASM-NEXT: lg %r1, 32(%r12)
# DISASM-NEXT: lg %r1, 40(%r12)
# DISASM-NEXT: lg %r1, 48(%r12)

# CHECK: Hex dump of section '.data':
# CHECK-NEXT: 00180020 00280030 00000018 00000020
# CHECK-NEXT: 00000028 00000030 00000000 00000018
# CHECK-NEXT: 00000000 00000020 00000000 00000028
# CHECK-NEXT: 00000000 00000030

.text
larl %r12, _GLOBAL_OFFSET_TABLE_
.reloc .+2, R_390_GOTENT, foo+2
larl %r1, 0
.reloc .+2, R_390_GOTENT, bar+2
larl %r1, 0
.reloc .+2, R_390_GOTPLTENT, foo+2
larl %r1, 0
.reloc .+2, R_390_GOTPLTENT, bar+2
larl %r1, 0
.reloc .+2, R_390_GOT12, foo
l %r1, 0(%r12)
.reloc .+2, R_390_GOT12, bar
l %r1, 0(%r12)
.reloc .+2, R_390_GOTPLT12, foo
l %r1, 0(%r12)
.reloc .+2, R_390_GOTPLT12, bar
l %r1, 0(%r12)
.reloc .+2, R_390_GOT20, foo
lg %r1, 0(%r12)
.reloc .+2, R_390_GOT20, bar
lg %r1, 0(%r12)
.reloc .+2, R_390_GOTPLT20, foo
lg %r1, 0(%r12)
.reloc .+2, R_390_GOTPLT20, bar
lg %r1, 0(%r12)

.data
.reloc ., R_390_GOT16, foo
.space 2
.reloc ., R_390_GOT16, bar
.space 2
.reloc ., R_390_GOTPLT16, foo
.space 2
.reloc ., R_390_GOTPLT16, bar
.space 2
.reloc ., R_390_GOT32, foo
.space 4
.reloc ., R_390_GOT32, bar
.space 4
.reloc ., R_390_GOTPLT32, foo
.space 4
.reloc ., R_390_GOTPLT32, bar
.space 4
.reloc ., R_390_GOT64, foo
.space 8
.reloc ., R_390_GOT64, bar
.space 8
.reloc ., R_390_GOTPLT64, foo
.space 8
.reloc ., R_390_GOTPLT64, bar
.space 8
