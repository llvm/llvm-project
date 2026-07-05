# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o %t.o -shared --emit-relocs -o %t.so
# RUN: llvm-readelf -S -r %t.so | FileCheck %s --check-prefixes=CHECK,RELOC
# RUN: llvm-dwarfdump --eh-frame %t.so | FileCheck %s --check-prefix=EH

## Also show that the merging happens when going via a -r link.
# RUN: ld.lld -r %t.o %t.o -o %t.ro
# RUN: ld.lld %t.ro -o %t2.so -shared
# RUN: llvm-readelf -S -r %t2.so | FileCheck %s

# CHECK:       Name      Type     Address              Off      Size   ES Flg Lk Inf Al
# CHECK:      .eh_frame  PROGBITS [[#%x,]]             [[#%x,]] 000058 00   A  0   0  8
# CHECK:      foo        PROGBITS {{0*}}[[#%x,FOO:]]   [[#%x,]] 000002 00  AX  0   0  1
# CHECK-NEXT: bar        PROGBITS {{0*}}[[#%x,FOO+2]]  [[#%x,]] 000002 00  AX  0   0  1

# RELOC:        Offset             Info     Type          Symbol's Value  Symbol's Name + Addend
# RELOC-NEXT: {{0*}}[[#%x,OFF:]]   [[#%x,]] R_X86_64_PC32 [[#%x,]]        foo + 0
# RELOC-NEXT: {{0*}}[[#%x,OFF+20]] [[#%x,]] R_X86_64_PC32 [[#%x,]]        bar + 0
# RELOC-NEXT: {{0*}}[[#OFF+40]]    [[#%x,]] R_X86_64_PC32 [[#%x,]]        foo + 1
# RELOC-NEXT: {{0*}}[[#%x,OFF-24]] [[#%x,]] R_X86_64_NONE 0{{$}}

# EH:          Format:                DWARF32
# EH:        00000018 00000010 0000001c FDE cie=00000000 pc={{0*}}[[#%x,FOO:]]...
# EH-SAME:   {{0*}}[[#%x,FOO+1]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:  
# EH:        0000002c 00000010 00000030 FDE cie=00000000 pc={{0*}}[[#%x,FOO+2]]...{{0*}}[[#%x,FOO+4]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:
# EH:        00000040 00000010 00000044 FDE cie=00000000 pc={{0*}}[[#%x,FOO+1]]...{{0*}}[[#%x,FOO+2]]
# EH-COUNT-3:  DW_CFA_nop:
# EH-EMPTY:
# EH-NEXT:     0x[[#%x,]]: CFA=RSP+8: RIP=[CFA-8]
# EH-EMPTY:
# EH-NEXT:   00000054 ZERO terminator

        .section	foo,"ax",@progbits
	.cfi_startproc
        nop
	.cfi_endproc

        .section	bar,"axG",@progbits,foo,comdat
        .cfi_startproc
        nop
        nop
	.cfi_endproc
