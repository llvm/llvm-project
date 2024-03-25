# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -soname=t.so -o %t.so

# RUN: llvm-readelf -S -s -x .data %t.so | FileCheck %s

# CHECK: Section Headers:
# CHECK: .plt PROGBITS 0000000000001290
# CHECK: .got PROGBITS 0000000000002390

# CHECK: Symbol table '.symtab'
# CHECK: 0000000000001288 {{.*}}  bar

## Note: foo is the first (and only) PLT entry, which resides at .plt + 32
## PLTOFF (foo) is (.plt + 32) - .got == 0x12b0 - 0x2390 == 0xffffef20
## GOTOFF (bar) is bar - .got == 0x1288 - 0x2390 == 0xffffeef8
# CHECK: Hex dump of section '.data':
# CHECK-NEXT: eef8ef20 ffffeef8 ffffef20 ffffffff
# CHECK-NEXT: ffffeef8 ffffffff ffffef20

bar:
  br %r14

.data
.reloc ., R_390_GOTOFF16, bar
.space 2
.reloc ., R_390_PLTOFF16, foo
.space 2
.reloc ., R_390_GOTOFF, bar
.space 4
.reloc ., R_390_PLTOFF32, foo
.space 4
.reloc ., R_390_GOTOFF64, bar
.space 8
.reloc ., R_390_PLTOFF64, foo
.space 8
