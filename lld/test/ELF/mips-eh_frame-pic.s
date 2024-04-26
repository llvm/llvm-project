# REQUIRES: mips
## Check that we can link a shared library containing an eh_frame section without
## -z notext. This was not possible LLVM started emitting values using the
## DW_EH_PE_pcrel | DW_EH_PE_sdata4 encoding.

## It should not be possible to link code compiled without -fPIC:
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t-nopic.o
# RUN: llvm-dwarfdump --eh-frame %t-nopic.o | FileCheck %s --check-prefix=ABS64-EH-FRAME
# RUN: llvm-readobj -r %t-nopic.o | FileCheck %s --check-prefixes=RELOCS,ABS64-RELOCS
# RUN: not ld.lld -shared %t-nopic.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOPIC-ERR
## Note: ld.bfd can link this file because it rewrites the .eh_frame section to use
## relative addressing.
# NOPIC-ERR: ld.lld: error: relocation R_MIPS_64 cannot be used against local symbol

## For -fPIC, .eh_frame should contain DW_EH_PE_pcrel | DW_EH_PE_sdata4 values:
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux --position-independent %s -o %t-pic.o
# RUN: llvm-readobj -r %t-pic.o | FileCheck %s --check-prefixes=RELOCS,PIC64-RELOCS
# RUN: ld.lld -shared %t-pic.o -o %t-pic.so
# RUN: llvm-dwarfdump --eh-frame %t-pic.so | FileCheck %s --check-prefix=PIC-EH-FRAME

## Also check MIPS32:
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-nopic32.o
# RUN: llvm-dwarfdump --eh-frame %t-nopic32.o | FileCheck %s --check-prefix=ABS32-EH-FRAME
# RUN: llvm-readobj -r %t-nopic32.o | FileCheck %s --check-prefixes=RELOCS,ABS32-RELOCS
# RUN: not ld.lld -shared %t-nopic32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOPIC32-ERR
## Note: ld.bfd can link this file because it rewrites the .eh_frame section to use
## relative addressing.
# NOPIC32-ERR: ld.lld: error: relocation R_MIPS_32 cannot be used against local symbol

## For -fPIC, .eh_frame should contain DW_EH_PE_pcrel | DW_EH_PE_sdata4 values:
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux --position-independent %s -o %t-pic32.o
# RUN: llvm-readobj -r %t-pic32.o | FileCheck %s --check-prefixes=RELOCS,PIC32-RELOCS
# RUN: ld.lld -shared %t-pic32.o -o %t-pic32.so
# RUN: llvm-dwarfdump --eh-frame %t-pic32.so | FileCheck %s --check-prefix=PIC-EH-FRAME

# RELOCS:            .rel{{a?}}.eh_frame {
# ABS32-RELOCS-NEXT:   0x1C R_MIPS_32 .text
# ABS64-RELOCS-NEXT:   0x1C R_MIPS_64/R_MIPS_NONE/R_MIPS_NONE .text
# PIC64-RELOCS-NEXT:   0x1C R_MIPS_PC32/R_MIPS_NONE/R_MIPS_NONE <null>
# PIC32-RELOCS-NEXT:   0x1C R_MIPS_PC32 <null>
# RELOCS-NEXT:       }

# ABS64-EH-FRAME: Augmentation data: 0C
##                                   ^^ fde pointer encoding: DW_EH_PE_sdata8
# ABS32-EH-FRAME: Augmentation data: 0B
##                                   ^^ fde pointer encoding: DW_EH_PE_sdata4
# PIC-EH-FRAME: Augmentation data: 1B
##                                 ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
## Note: ld.bfd converts the R_MIPS_64 relocs to DW_EH_PE_pcrel | DW_EH_PE_sdata8
## for N64 ABI (and DW_EH_PE_pcrel | DW_EH_PE_sdata4 for MIPS32)

.ent func
.global func
func:
	.cfi_startproc
	nop
	.cfi_endproc
.end func
