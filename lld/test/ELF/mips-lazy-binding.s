# REQUIRES: mips
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=mips %t/calls.s -o %t/mips.o
# RUN: llvm-mc -filetype=obj -triple=mipsel %t/calls.s -o %t/mipsel.o
# RUN: llvm-mc -filetype=obj -triple=mips64 -defsym=ELF64=1 %t/calls.s -o %t/mips64.o
# RUN: llvm-mc -filetype=obj -triple=mips64el -defsym=ELF64=1 %t/calls.s -o %t/mips64el.o
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/mips.o -o %t/mips.so
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/mipsel.o -o %t/mipsel.so
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/mips64.o -o %t/mips64.so
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/mips64el.o -o %t/mips64el.so
# RUN: llvm-readelf --dyn-syms -A %t/mips.so | FileCheck %s --check-prefixes=SYMS,GOT32
# RUN: llvm-readelf --dyn-syms -A %t/mipsel.so | FileCheck %s --check-prefixes=SYMS,GOT32
# RUN: llvm-readelf --dyn-syms -A %t/mips64.so | FileCheck %s --check-prefixes=SYMS,GOT64
# RUN: llvm-readelf --dyn-syms -A %t/mips64el.so | FileCheck %s --check-prefixes=SYMS,GOT64
# RUN: llvm-objdump -s -j .MIPS.stubs %t/mips.so | FileCheck %s --check-prefix=BE32
# RUN: llvm-objdump -s -j .MIPS.stubs %t/mipsel.so | FileCheck %s --check-prefix=LE32
# RUN: llvm-objdump -s -j .MIPS.stubs %t/mips64.so | FileCheck %s --check-prefix=BE64
# RUN: llvm-objdump -s -j .MIPS.stubs %t/mips64el.so | FileCheck %s --check-prefix=LE64
# RUN: ld.lld -shared -z now --section-start=.MIPS.stubs=0x20000 %t/mips.o -o %t/now.so
# RUN: llvm-objdump -s -j .MIPS.stubs %t/now.so | FileCheck %s --check-prefix=BE32

## Imported call targets keep SHN_UNDEF, but their dynamic values and primary GOT
## slots point at lazy stubs. An untyped undefined call target becomes STT_FUNC.
# SYMS: Symbol table '.dynsym'
# SYMS: 2: {{0*}}20000 0 FUNC GLOBAL DEFAULT UND typed
# SYMS: 3: {{0*}}20010 0 FUNC GLOBAL DEFAULT UND untyped
# GOT32: Primary GOT:
# GOT32: Global entries:
# GOT32-NEXT: Address Access Initial Sym.Val. Type Ndx Name
# GOT32-NEXT: 00030008 -32744(gp) 00020000 00020000 FUNC UND typed
# GOT32-NEXT: 0003000c -32740(gp) 00020010 00020010 FUNC UND untyped
# GOT64: Primary GOT:
# GOT64: Global entries:
# GOT64-NEXT: Address Access Initial Sym.Val. Type Ndx Name
# GOT64-NEXT: 0000000000030010 -32736(gp) 0000000000020000 0000000000020000 FUNC UND typed
# GOT64-NEXT: 0000000000030018 -32728(gp) 0000000000020010 0000000000020010 FUNC UND untyped

## Each normal stub loads the resolver from GOT[0], preserves $ra in $15, calls
## the resolver, and places its dynamic symbol index in $24 in the delay slot.
# BE32: Contents of section .MIPS.stubs:
# BE32-NEXT: 20000 8f998010 03e07825 0320f809 24180002
# BE32-NEXT: 20010 8f998010 03e07825 0320f809 24180003
# LE32: Contents of section .MIPS.stubs:
# LE32-NEXT: 20000 1080998f 2578e003 09f82003 02001824
# LE32-NEXT: 20010 1080998f 2578e003 09f82003 03001824
# BE64: Contents of section .MIPS.stubs:
# BE64-NEXT: 20000 df998010 03e07825 0320f809 64180002
# BE64-NEXT: 20010 df998010 03e07825 0320f809 64180003
# LE64: Contents of section .MIPS.stubs:
# LE64-NEXT: 20000 108099df 2578e003 09f82003 02001864
# LE64-NEXT: 20010 108099df 2578e003 09f82003 03001864

# RUN: ld.lld -shared --section-start=.MIPS.stubs=0x20000 -z hazardplt %t/mips.o -o %t/hazard32.so
# RUN: ld.lld -shared --section-start=.MIPS.stubs=0x20000 -z hazardplt %t/mips64el.o -o %t/hazard64.so
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn -j .MIPS.stubs %t/hazard32.so | FileCheck %s --check-prefixes=HAZARD,HAZARD32
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn -j .MIPS.stubs %t/hazard64.so | FileCheck %s --check-prefixes=HAZARD,HAZARD64
# HAZARD32: 20000: lw $25, -32752($gp)
# HAZARD64: 20000: ld $25, -32752($gp)
# HAZARD-NEXT: 20004: move $15, $ra
# HAZARD-NEXT: 20008: jalr.hb $25
# HAZARD32-NEXT: 2000c: addiu $24, $zero, 2
# HAZARD64-NEXT: 2000c: daddiu $24, $zero, 2

## Functions provided by a DSO also use lazy stubs when linking an executable.
# RUN: llvm-mc -filetype=obj -triple=mips %t/definitions.s -o %t/definitions.o
# RUN: ld.lld -shared %t/definitions.o -o %t/provider.so
# RUN: ld.lld -e caller --section-start=.MIPS.stubs=0x20000 %t/mips.o %t/provider.so -o %t/exe
# RUN: llvm-readelf --dyn-syms %t/exe | FileCheck %s --check-prefix=EXE
# EXE: 1: 00020000 0 FUNC GLOBAL DEFAULT UND typed
# EXE: 2: 00020010 0 FUNC GLOBAL DEFAULT UND untyped

## Address references and definitions must not acquire lazy stubs, even when the
## same symbol also has call relocations. Only "lazy" gets a stub in this output.
# RUN: llvm-mc -filetype=obj -triple=mips %t/mixed.s -o %t/mixed.o
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 %t/mixed.o -o %t/mixed.so
# RUN: llvm-readelf -S --dyn-syms %t/mixed.so | FileCheck %s --check-prefix=MIXED
# MIXED: .MIPS.stubs PROGBITS 00020000 {{[0-9a-f]+}} 000010
# MIXED: Symbol table '.dynsym'
# MIXED-DAG: 00020000 0 FUNC GLOBAL DEFAULT UND lazy
# MIXED-DAG: 00000000 0 FUNC GLOBAL DEFAULT UND addressed
# MIXED-DAG: 00010020 0 FUNC GLOBAL DEFAULT {{[0-9]+}} defined

## An address reference excludes a call target regardless of input scan order.
## The other call-only target remains lazy and occupies the only stub slot.
# RUN: llvm-mc -filetype=obj -triple=mips %t/address.s -o %t/address.o
# RUN: ld.lld -shared --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/address.o %t/mips.o -o %t/address-first.so
# RUN: ld.lld -shared --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 %t/mips.o %t/address.o -o %t/calls-first.so
# RUN: llvm-readelf -S --dyn-syms -A %t/address-first.so | FileCheck %s --check-prefix=ORDER
# RUN: llvm-readelf -S --dyn-syms -A %t/calls-first.so | FileCheck %s --check-prefix=ORDER
# ORDER: .MIPS.stubs PROGBITS 00020000 {{[0-9a-f]+}} 000010
# ORDER: Symbol table '.dynsym'
# ORDER-DAG: 00000000 0 FUNC GLOBAL DEFAULT UND typed
# ORDER-DAG: 00020000 0 FUNC GLOBAL DEFAULT UND untyped
# ORDER: Primary GOT:
# ORDER: Global entries:
# ORDER-NEXT: Address Access Initial Sym.Val. Type Ndx Name
# ORDER-NEXT: 00030008 -32744(gp) 00000000 00000000 FUNC UND typed
# ORDER-NEXT: 0003000c -32740(gp) 00020000 00020000 FUNC UND untyped

## A reference from a live .eh_frame CIE personality pointer requires eager
## binding. Use -z notext for that absolute pointer, independently of the stub.
# RUN: llvm-mc -filetype=obj -triple=mips --position-independent %t/personality.s -o %t/personality.o
# RUN: ld.lld -shared -z notext %t/personality.o -o %t/personality.so
# RUN: llvm-readelf -S --dyn-syms %t/personality.so | FileCheck %s --check-prefix=PERSONALITY --implicit-check-not=.MIPS.stubs
# PERSONALITY: Symbol table '.dynsym'
# PERSONALITY: 00000000 0 FUNC GLOBAL DEFAULT UND personality

## A noncanonical _gp does not provide the resolver's primary GOT context.
# RUN: ld.lld -shared --defsym=_gp=0x30080 --section-start=.got=0x30000 %t/mips.o -o %t/custom-gp.so
# RUN: llvm-readelf --dyn-syms -A %t/custom-gp.so | FileCheck %s --check-prefix=CUSTOM-GP
# CUSTOM-GP: Symbol table '.dynsym'
# CUSTOM-GP-DAG: 00000000 0 FUNC GLOBAL DEFAULT UND typed
# CUSTOM-GP-DAG: 00000000 0 FUNC GLOBAL DEFAULT UND untyped
# CUSTOM-GP: Primary GOT:
# CUSTOM-GP: Global entries:
# CUSTOM-GP-NEXT: Address Access Initial Sym.Val. Type Ndx Name
# CUSTOM-GP-NEXT: 00030008 {{-[0-9]+}}(gp) 00000000 00000000 FUNC UND typed
# CUSTOM-GP-NEXT: 0003000c {{-[0-9]+}}(gp) 00000000 00000000 FUNC UND untyped

## An explicitly defined canonical _gp supports lazy binding as usual.
# RUN: ld.lld -shared -Ttext=0x10000 --section-start=.MIPS.stubs=0x20000 --section-start=.got=0x30000 --defsym=_gp=0x37ff0 %t/mips.o -o %t/canonical-gp.so
# RUN: llvm-readelf --dyn-syms -A %t/canonical-gp.so | FileCheck %s --check-prefixes=SYMS,GOT32

## A GP-relative section before .got can also give the default _gp a different
## base. Keep eager GOT/dynsym values even if stub space was already reserved.
# RUN: llvm-mc -filetype=obj -triple=mips %t/gp-layout.s -o %t/gp-layout.o
# RUN: ld.lld -shared --script=%t/gp-layout.lds %t/gp-layout.o -o %t/gp-layout.so
# RUN: llvm-readelf -s -A %t/gp-layout.so | FileCheck %s --check-prefix=GP-LAYOUT
# GP-LAYOUT: Symbol table '.dynsym'
# GP-LAYOUT: 00000000 0 FUNC GLOBAL DEFAULT UND target
# GP-LAYOUT: Symbol table '.symtab'
# GP-LAYOUT: 00027ff0 {{.*}} _gp
# GP-LAYOUT: Primary GOT:
# GP-LAYOUT: Global entries:
# GP-LAYOUT-NEXT: Address Access Initial Sym.Val. Type Ndx Name
# GP-LAYOUT-NEXT: 00020018 {{-[0-9]+}}(gp) 00000000 00000000 FUNC UND target

## Relocatable links, static links, and unsupported microMIPS links have no stubs.
# RUN: ld.lld -r %t/mips.o -o %t/relocatable.o
# RUN: llvm-readelf -S %t/relocatable.o | FileCheck %s --check-prefix=NO-STUB --implicit-check-not=.MIPS.stubs
# RUN: ld.lld -e caller %t/mips.o %t/definitions.o -o %t/static
# RUN: llvm-readelf -S %t/static | FileCheck %s --check-prefix=NO-STUB --implicit-check-not=.MIPS.stubs
# RUN: llvm-mc -filetype=obj -triple=mips -mattr=micromips -defsym=MICRO=1 %t/calls.s -o %t/micro.o
# RUN: ld.lld -shared %t/micro.o -o %t/micro.so
# RUN: llvm-readelf -S %t/micro.so | FileCheck %s --check-prefix=NO-STUB --implicit-check-not=.MIPS.stubs
# NO-STUB: Section Headers:

#--- calls.s
.set noreorder
.text
.globl caller
caller:
.ifdef ELF64
  ld $25, %call16(typed)($gp)
.else
  lw $25, %call16(typed)($gp)
.endif
.Ltyped_call:
  jalr $25
.ifdef MICRO
  .reloc .Ltyped_call, R_MICROMIPS_JALR, typed
.else
  .reloc .Ltyped_call, R_MIPS_JALR, typed
.endif
  nop
.ifdef ELF64
  ld $25, %call16(untyped)($gp)
.else
  lw $25, %call16(untyped)($gp)
.endif
  jalr $25
  nop
  lui $2, %call_hi(typed)
.ifdef ELF64
  ld $25, %call_lo(typed)($2)
.else
  lw $25, %call_lo(typed)($2)
.endif
.globl typed, untyped
.type typed, @function

#--- definitions.s
.text
.globl typed, untyped
.type typed, @function
.type untyped, @function
typed:
  nop
untyped:
  nop

#--- mixed.s
.set noreorder
.text
.globl caller
caller:
  lw $25, %call16(lazy)($gp)
  lw $25, %call16(addressed)($gp)
  lw $25, %call16(defined)($gp)
  lw $25, %call16(hidden)($gp)
  nop
  nop
  nop
  nop
.globl lazy, addressed, defined, hidden
.type lazy, @function
.type addressed, @function
.type defined, @function
.type hidden, @function
defined:
  nop
.hidden hidden
hidden:
  nop
.data
  .word addressed

#--- address.s
.data
  .word typed

#--- personality.s
.set noreorder
.text
.globl caller
caller:
  .cfi_startproc
  .cfi_personality 0, personality
  lw $25, %call16(personality)($gp)
  jalr $25
  nop
  .cfi_endproc
.globl personality
.type personality, @function

#--- gp-layout.s
.text
.globl caller
caller:
  lw $25, %call16(target)($gp)
.globl target
.type target, @function
.sdata
  .word 0

#--- gp-layout.lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .sdata 0x20000 : { *(.sdata) }
  .got 0x20010 : { *(.got) }
}
