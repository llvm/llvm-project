# REQUIRES: mips
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=mips %t/primary.s -o %t/primary.o
# RUN: llvm-mc -filetype=obj -triple=mips %t/secondary.s -o %t/secondary.o
# RUN: ld.lld -shared -mips-got-size=16 --section-start=.MIPS.stubs=0x10000 \
# RUN:   --section-start=.got=0x20000 %t/primary.o %t/secondary.o -o %t/multi.so
# RUN: llvm-readobj --dyn-syms -r -A %t/multi.so | FileCheck %s --check-prefix=MULTI
# RUN: llvm-objdump -s -j .got -j .MIPS.stubs %t/multi.so | FileCheck %s --check-prefix=CONTENTS

## A symbol used by a secondary GOT cannot use the primary GOT's resolver even
## when the symbol also has a normal global entry in the primary GOT.
# MULTI:      Relocations [
# MULTI:        0x20014 R_MIPS_REL32 both
# MULTI-NEXT:   0x20018 R_MIPS_REL32 secondary
# MULTI:      DynamicSymbols [
# MULTI:        Name: primary
# MULTI-NEXT:   Value: 0x10000
# MULTI:        Section: Undefined
# MULTI:        Name: both
# MULTI-NEXT:   Value: 0x0
# MULTI:        Section: Undefined
# MULTI:        Name: secondary
# MULTI-NEXT:   Value: 0x0
# MULTI:        Section: Undefined
# MULTI:      Global entries [
# MULTI:        Initial: 0x10000
# MULTI-NEXT:   Value: 0x10000
# MULTI:        Name: primary
# MULTI:        Initial: 0x0
# MULTI-NEXT:   Value: 0x0
# MULTI:        Name: both
# MULTI:        Initial: 0x0
# MULTI-NEXT:   Value: 0x0
# MULTI:        Name: secondary

# CONTENTS:      Contents of section .MIPS.stubs:
# CONTENTS-NEXT: 10000 8f998010 03e07825 0320f809 24180001
# CONTENTS:      Contents of section .got:
# CONTENTS-NEXT: 20000 00000000 80000000 00010000 00000000
# CONTENTS-NEXT: 20010 00000000 00000000 00000000

## Include .MIPS.stubs in a section referenced by local GOT page relocations.
## Its size changes the conservative page count from two to three entries.
# RUN: llvm-mc -filetype=obj -triple=mips %t/pages.s -o %t/pages.o
# RUN: ld.lld -shared -T %t/pages.lds %t/pages.o -o %t/pages.so
# RUN: llvm-readobj -A %t/pages.so | FileCheck %s --check-prefix=PAGES
# RUN: llvm-readelf --dyn-syms %t/pages.so | FileCheck %s --check-prefix=PAGESTUB

# PAGES:      Local entries [
# PAGES-NEXT:   Entry {
# PAGES-NEXT:     Address:
# PAGES-NEXT:     Access:
# PAGES-NEXT:     Initial: 0x10000
# PAGES-NEXT:   }
# PAGES-NEXT:   Entry {
# PAGES-NEXT:     Address:
# PAGES-NEXT:     Access:
# PAGES-NEXT:     Initial: 0x20000
# PAGES-NEXT:   }
# PAGES-NEXT:   Entry {
# PAGES-NEXT:     Address:
# PAGES-NEXT:     Access:
# PAGES-NEXT:     Initial: 0x30000
# PAGES-NEXT:   }
# PAGES-NEXT: ]
# PAGES:        Initial: 0x1FFF0
# PAGES-NEXT:   Value: 0x1FFF0
# PAGES:        Name: target
# PAGESTUB: {{[0-9]+}}: 0001fff0 0 FUNC GLOBAL DEFAULT UND target

#--- primary.s
.text
.globl primary, both
.type primary,@function
.type both,@function
lw $25, %call16(primary)($gp)
lw $25, %call16(both)($gp)

#--- secondary.s
.text
.globl both, secondary
.type both,@function
.type secondary,@function
lw $25, %call16(both)($gp)
lw $25, %call16(secondary)($gp)

#--- pages.s
.text
.globl target
.type target,@function
local:
lw $2, %got(local)($gp)
lw $25, %call16(target)($gp)
.space 0xfff0 - (. - local)

#--- pages.lds
SECTIONS {
  .text 0x10000 : { *(.text) *(.MIPS.stubs) }
  .got 0x30000 : { *(.got) }
}
