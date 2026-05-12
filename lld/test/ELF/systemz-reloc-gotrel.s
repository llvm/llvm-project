# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -soname=t.so -o %t.so

# RUN: llvm-readelf -S -s -x .data -x .data1 %t.so | FileCheck %s

# CHECK: Section Headers:
# CHECK: .plt PROGBITS 0000000000001290
# CHECK: .got PROGBITS 0000000000002390

# CHECK: Symbol table '.symtab'
# CHECK: 0000000000001288 {{.*}}  bar
# CHECK: 000000000000128a {{.*}}  baz

## .data holds relocations against non-preemptible symbols (link-time
## constants): GOTOFF (bar) and PLTOFF (baz). For PLTOFF (baz) the scan
## rewrites R_PLT_GOTREL → R_GOTREL via fromPlt so the result is
## baz - .got, not the offset of a PLT entry.
## GOTOFF (bar) = bar - .got = 0x1288 - 0x2390 = 0xffffeef8
## PLTOFF (baz) = baz - .got = 0x128a - 0x2390 = 0xffffeefa
# CHECK: Hex dump of section '.data':
# CHECK-NEXT: eef8eefa ffffeef8 ffffeefa ffffffff
# CHECK-NEXT: ffffeef8 ffffffff ffffeefa

## .data1 holds relocations against the preemptible foo, which uses the
## first (and only) PLT entry at .plt + 32.
## PLTOFF (foo) = (.plt + 32) - .got = 0x12b0 - 0x2390 = 0xffffef20
# CHECK: Hex dump of section '.data1':
# CHECK-NEXT: ef20ffff ef20ffff ffffffff ef20

bar:
  br %r14

.hidden baz
.globl baz
baz:
  br %r14

.data
.reloc ., R_390_GOTOFF16, bar
.space 2
.reloc ., R_390_PLTOFF16, baz
.space 2
.reloc ., R_390_GOTOFF, bar
.space 4
.reloc ., R_390_PLTOFF32, baz
.space 4
.reloc ., R_390_GOTOFF64, bar
.space 8
.reloc ., R_390_PLTOFF64, baz
.space 8

.section .data1,"aw"
.reloc ., R_390_PLTOFF16, foo
.space 2
.reloc ., R_390_PLTOFF32, foo
.space 4
.reloc ., R_390_PLTOFF64, foo
.space 8
