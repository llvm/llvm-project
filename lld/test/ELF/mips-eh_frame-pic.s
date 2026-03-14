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
# RUN: llvm-dwarfdump --eh-frame %t-pic.so | FileCheck %s --check-prefix=PIC64-EH-FRAME

## Also check MIPS32:
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-nopic32.o
# RUN: llvm-dwarfdump --eh-frame %t-nopic32.o | FileCheck %s --check-prefix=ABS32-EH-FRAME
# RUN: llvm-readobj -r %t-nopic32.o | FileCheck %s --check-prefixes=RELOCS,ABS32-RELOCS
# RUN: not ld.lld -shared %t-nopic32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOPIC32-ERR
## Note: ld.bfd can link this file because it rewrites the .eh_frame section to use
## relative addressing.
# NOPIC32-ERR: ld.lld: error: relocation R_MIPS_32 cannot be used against local symbol

## https://github.com/llvm/llvm-project/issues/88852: getFdePc should return a
## 32-bit address.
# RUN: ld.lld --eh-frame-hdr -Ttext=0x80000000 %t-nopic32.o -o %t-nopic32
# RUN: llvm-readelf -x .eh_frame_hdr %t-nopic32 | FileCheck %s --check-prefix=NOPIC32-HDR

## For -fPIC, .eh_frame should contain DW_EH_PE_pcrel | DW_EH_PE_sdata4 values:
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux --position-independent %s -o %t-pic32.o
# RUN: llvm-readobj -r %t-pic32.o | FileCheck %s --check-prefixes=RELOCS,PIC32-RELOCS
# RUN: ld.lld -shared %t-pic32.o -o %t-pic32.so
# RUN: llvm-dwarfdump --eh-frame %t-pic32.so | FileCheck %s --check-prefix=PIC32-EH-FRAME

# RELOCS:            .rel{{a?}}.eh_frame {
# ABS32-RELOCS-NEXT:   0x1C R_MIPS_32 .text
# ABS64-RELOCS-NEXT:   0x1C R_MIPS_64/R_MIPS_NONE/R_MIPS_NONE .text
# PIC64-RELOCS-NEXT:   0x1C R_MIPS_PC32/R_MIPS_NONE/R_MIPS_NONE .L0
# PIC32-RELOCS-NEXT:   0x1C R_MIPS_PC32 .L0
# RELOCS-NEXT:       }

# ABS64-EH-FRAME: Augmentation data: 0C
##                                   ^^ fde pointer encoding: DW_EH_PE_sdata8
# ABS32-EH-FRAME: Augmentation data: 0B
##                                   ^^ fde pointer encoding: DW_EH_PE_sdata4
# PIC32-EH-FRAME: Augmentation data: 1B
##                                 ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
# PIC64-EH-FRAME: Augmentation data: 1B
##                                 ^^ fde pointer encoding: DW_EH_PE_pcrel | DW_EH_PE_sdata4
## Note: ld.bfd converts the R_MIPS_64 relocs to DW_EH_PE_pcrel | DW_EH_PE_sdata8
## for N64 ABI (and DW_EH_PE_pcrel | DW_EH_PE_sdata4 for MIPS32)

# NOPIC32-HDR: Hex dump of section '.eh_frame_hdr':
# NOPIC32-HDR: 0x80010038 011b033b 00000010 00000001 fffeffc8 .
# NOPIC32-HDR: 0x80010048 00000028                            .

.ent func
.global func
func:
	.cfi_startproc
	nop
	.cfi_endproc
.end func
