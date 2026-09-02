## The TType references to _OBJC_EHTYPE_$_NSException are PC-relative
## pointer-to-GOT relocations: the assembler leaves a meaningless
## site-dependent value in each TType slot, so identical exception tables
## differ in their raw bytes and can only be folded by ICF after the relocated
## fields are normalized. Reduced from a real-world ObjC object.
##
## _f0 and _f1 are unwound via compact unwind. Their bodies and (normalized)
## tables are identical, so the functions fold together with their tables.
##
## _g0 and _g1 carry a lone .cfi_offset with no frame, which compact unwind
## cannot encode, so they get DWARF-mode entries and are unwound via their
## FDEs. Their bodies differ, so only their tables fold; the FDE's LSDA
## pointer is resolved when the FDE is parsed and must be canonicalized after
## ICF, or their __unwind_info LSDA entries would point at the folded-away
## tables.

# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %s -o %t.o
# RUN: llvm-objdump --macho --reloc --section=__gcc_except_tab %t.o | \
# RUN:   FileCheck %s --check-prefix=INPUT
# RUN: %lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -undefined dynamic_lookup --icf=all %t.o -o %t
# RUN: llvm-objdump --macho --syms --unwind-info --dwarf=frames %t | \
# RUN:   FileCheck %s

## Sanity check: the TType entries are relocations, so the tables can only be
## folded after normalization.
# INPUT:         Relocation information (__TEXT,__gcc_except_tab) 4 entries
# INPUT-COUNT-4: PTRTGOT False     _OBJC_EHTYPE_$_NSException

# CHECK-LABEL: SYMBOL TABLE:
# CHECK: [[#%x,LSDA:]] l   O __TEXT,__gcc_except_tab GCC_except_table0
# CHECK: [[#LSDA]]     l   O __TEXT,__gcc_except_tab GCC_except_table1
# CHECK: [[#LSDA]]     l   O __TEXT,__gcc_except_tab GCC_except_table2
# CHECK: [[#LSDA]]     l   O __TEXT,__gcc_except_tab GCC_except_table3
# CHECK: [[#%x,F0:]]   g   F __TEXT,__text _f0
# CHECK: [[#F0]]       g   F __TEXT,__text _f1
# CHECK: [[#%x,G0:]]   g   F __TEXT,__text _g0
# CHECK: [[#%x,G1:]]   g   F __TEXT,__text _g1

# CHECK: LSDA descriptors:
# CHECK-NEXT: [0]: function offset=0x[[#%.8x,F0]], LSDA offset=0x[[#%.8x,LSDA]]
# CHECK-NEXT: [1]: function offset=0x[[#%.8x,G0]], LSDA offset=0x[[#%.8x,LSDA]]
# CHECK-NEXT: [2]: function offset=0x[[#%.8x,G1]], LSDA offset=0x[[#%.8x,LSDA]]

# CHECK-LABEL: .eh_frame contents:
# CHECK: FDE cie={{.+}} pc=[[#%.8x,G0]]...{{.+}}
# CHECK: LSDA Address: [[#%.16x,LSDA]]
# CHECK: FDE cie={{.+}} pc=[[#%.8x,G1]]...{{.+}}
# CHECK: LSDA Address: [[#%.16x,LSDA]]

.section __TEXT,__text,regular,pure_instructions
.globl _f0, _f1, _g0, _g1
.p2align 2
_f0:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, GCC_except_table0
  ret
  .cfi_endproc

_f1:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, GCC_except_table1
  ret
  .cfi_endproc

## The lone .cfi_offset forces a DWARF-mode unwind entry: compact unwind can
## only encode callee-saved registers spilled in adjacent pairs below a frame.
_g0:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, GCC_except_table2
  .cfi_offset w19, -24
  mov w0, #0
  ret
  .cfi_endproc

_g1:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, GCC_except_table3
  .cfi_offset w19, -24
  mov w0, #1
  ret
  .cfi_endproc

.section __TEXT,__gcc_except_tab
.p2align 2
GCC_except_table0:
  .long _OBJC_EHTYPE_$_NSException@GOT-.
GCC_except_table1:
  .long _OBJC_EHTYPE_$_NSException@GOT-.
GCC_except_table2:
  .long _OBJC_EHTYPE_$_NSException@GOT-.
GCC_except_table3:
  .long _OBJC_EHTYPE_$_NSException@GOT-.

.subsections_via_symbols
