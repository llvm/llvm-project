# REQUIRES: x86
## When a .ctors section is moved to .init_array and its entries are reversed,
## every relocation must follow its entry: dynamic relocations (RELATIVE,
## symbolic, RELR), --emit-relocs output, REL (implicit addend) and CREL input.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 --crel a.s -o crel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 --crel crel-only.s -o crel-only.o
# RUN: llvm-mc -filetype=obj -triple=i386 a32.s -o a32.o

## .init_array is [gf, f5, f6] (.ctors reversed) followed by [f4].
# RUN: ld.lld -pie a.o -o pie
# RUN: llvm-nm pie > pie.txt
# RUN: llvm-readelf -S -r pie >> pie.txt
# RUN: FileCheck %s --check-prefix=PIE < pie.txt

# PIE:      [[#%x,F4:]] t f4
# PIE-NEXT: [[#%x,F5:]] t f5
# PIE-NEXT: [[#%x,F6:]] t f6
# PIE-NEXT: [[#%x,GF:]] T gf
# PIE:      .init_array INIT_ARRAY [[#%x,IA:]]
# PIE:      [[#%.16x,IA]]    {{.*}} R_X86_64_RELATIVE [[#%x,GF]]
# PIE-NEXT: [[#%.16x,IA+8]]  {{.*}} R_X86_64_RELATIVE [[#%x,F5]]
# PIE-NEXT: [[#%.16x,IA+16]] {{.*}} R_X86_64_RELATIVE [[#%x,F6]]
# PIE-NEXT: [[#%.16x,IA+24]] {{.*}} R_X86_64_RELATIVE [[#%x,F4]]

## In a shared object, the preemptible gf gets a symbolic relocation.
# RUN: ld.lld -shared a.o -o so
# RUN: llvm-readelf -S -r so | FileCheck %s --check-prefix=SHARED

# SHARED:      .init_array INIT_ARRAY [[#%x,IA:]]
# SHARED-DAG:  [[#%.16x,IA]]    {{.*}} R_X86_64_64 {{.*}} gf + 0
# SHARED-DAG:  [[#%.16x,IA+8]]  {{.*}} R_X86_64_RELATIVE
# SHARED-DAG:  [[#%.16x,IA+16]] {{.*}} R_X86_64_RELATIVE
# SHARED-DAG:  [[#%.16x,IA+24]] {{.*}} R_X86_64_RELATIVE

## RELR stores the addends in the section contents: [gf, f5, f6, f4].
# RUN: ld.lld -pie -z pack-relative-relocs a.o -o relr
# RUN: llvm-nm relr > relr.txt
# RUN: llvm-readelf -S -r -x .init_array relr >> relr.txt
# RUN: FileCheck %s --check-prefix=RELR < relr.txt

# RELR:      [[#%x,F4:]] t f4
# RELR-NEXT: [[#%x,F5:]] t f5
# RELR-NEXT: [[#%x,F6:]] t f6
# RELR-NEXT: [[#%x,GF:]] T gf
# RELR:      .init_array INIT_ARRAY [[#%x,IA:]]
# RELR:      Relocation section '.relr.dyn'
# RELR:      [[#%.16x,IA]] [[#%.16x,IA]]
# RELR:      Hex dump of section '.init_array':
# RELR-NEXT: 0x{{[0-9a-f]+}} 24120000 00000000 22120000 00000000
# RELR-NEXT: 0x{{[0-9a-f]+}} 23120000 00000000 21120000 00000000

## --emit-relocs copies the rewritten relocations, for RELA and CREL input.
# RUN: ld.lld --emit-relocs a.o -o emit
# RUN: llvm-readelf -S -r emit | FileCheck %s --check-prefix=EMIT
# RUN: ld.lld --emit-relocs crel.o -o emit-crel
# RUN: llvm-readelf -S -r emit-crel | FileCheck %s --check-prefix=EMIT

# EMIT:      .init_array INIT_ARRAY [[#%x,IA:]]
# EMIT:      Relocation section '.{{rela|crel}}.init_array'
# EMIT:      [[#%.16x,IA]]    {{.*}} R_X86_64_64 {{.*}} gf + 0
# EMIT-NEXT: [[#%.16x,IA+8]]  {{.*}} R_X86_64_64 {{.*}} .text + 2
# EMIT-NEXT: [[#%.16x,IA+16]] {{.*}} R_X86_64_64 {{.*}} .text + 3
# EMIT-NEXT: [[#%.16x,IA+24]] {{.*}} R_X86_64_64 {{.*}} .text + 1

## If the only relocations for .init_array come from the reversed .ctors,
## preserve the SHT_CREL output type and re-encode the rewritten relocations.
# RUN: ld.lld --emit-relocs crel-only.o -o emit-crel-only
# RUN: llvm-readelf -S -r emit-crel-only | FileCheck %s --check-prefix=CREL-ONLY

# CREL-ONLY:      .init_array       INIT_ARRAY      [[#%x,IA:]]
# CREL-ONLY:      .crel.init_array  CREL
# CREL-ONLY:      Relocation section '.crel.init_array'
# CREL-ONLY:      [[#%.16x,IA]]   {{.*}} R_X86_64_64 {{.*}} f1 + 0
# CREL-ONLY-NEXT: [[#%.16x,IA+8]] {{.*}} R_X86_64_64 {{.*}} f2 + 0

## Without --emit-relocs, CREL input gives the same output as RELA input.
# RUN: ld.lld -pie crel.o -o crel
# RUN: llvm-readelf -r -x .init_array pie > pie.dump
# RUN: llvm-readelf -r -x .init_array crel > crel.dump
# RUN: diff pie.dump crel.dump

## REL input: the implicit addends move with their entries, giving
## [f4+3, f4+2, f4+1, _start] where f4 = _start+1.
# RUN: ld.lld a32.o -o a32
# RUN: llvm-readelf -x .init_array a32 | FileCheck %s --check-prefix=REL

# REL:      Hex dump of section '.init_array':
# REL-NEXT: 0x{{[0-9a-f]+}} f8104000 f7104000 f6104000 f4104000

#--- a.s
.globl _start, gf
.text
_start:
  nop
f4:
  nop
f5:
  nop
f6:
  nop
gf:
  ret

.section .ctors, "aw", @progbits
.p2align 3
  .quad f6
  .quad f5
  .quad gf

.section .init_array, "aw", @init_array
.p2align 3
  .quad f4

#--- crel-only.s
.globl _start, f1, f2
.text
_start:
  nop
f1:
  nop
f2:
  ret

.section .ctors, "aw", @progbits
.p2align 3
  .quad f2
  .quad f1

## A zero-sized array section is sufficient to select the modern ABI, but does
## not contribute a relocation section that could mask the CREL output type.
.section .init_array, "aw", @init_array

#--- a32.s
.globl _start
.text
_start:
  nop
f4:
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop

.section .ctors, "aw", @progbits
  .long f4+1
  .long f4+2
  .long f4+3

.section .init_array, "aw", @init_array
  .long _start
