# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 %s -o %t/eh-frame-x86_64.o
# RUN: %lld -lSystem -lc++ %t/eh-frame-x86_64.o -o %t/eh-frame-x86_64
# RUN: llvm-objdump --macho --syms --indirect-symbols --unwind-info \
# RUN:   --dwarf=frames %t/eh-frame-x86_64 | FileCheck %s -D#BASE=0x100000000 -D#DWARF_ENC=4
# RUN: llvm-nm -m %t/eh-frame-x86_64 | FileCheck %s --check-prefix NO-EH-SYMS
# RUN: llvm-readobj --section-headers %t/eh-frame-x86_64 | FileCheck %s --check-prefix=ALIGN -D#ALIGN=3

## Test that we correctly handle the output of `ld -r`, which emits EH frames
## using subtractor relocations instead of implicitly encoding the offsets.
## In order to keep this test cross-platform, we check in ld64's output rather
## than invoking ld64 directly. NOTE: whenever this test is updated, the
## checked-in copy of `ld -r`'s output should be updated too!
# COM: ld -r %t/eh-frame-x86_64.o -o %S/Inputs/eh-frame-x86_64-r.o
# RUN: %lld -lSystem -lc++ %S/Inputs/eh-frame-x86_64-r.o -o %t/eh-frame-x86_64-r
# RUN: llvm-objdump --macho --syms --indirect-symbols --unwind-info \
# RUN:   --dwarf=frames %t/eh-frame-x86_64-r | FileCheck %s -D#BASE=0x100000000 -D#DWARF_ENC=4
# RUN: llvm-nm -m %t/eh-frame-x86_64-r | FileCheck %s --check-prefix NO-EH-SYMS
# RUN: llvm-readobj --section-headers %t/eh-frame-x86_64-r | FileCheck %s --check-prefix=ALIGN -D#ALIGN=3

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11.0 %s -o %t/eh-frame-arm64.o
# RUN: %lld -arch arm64 -lSystem -lc++ %t/eh-frame-arm64.o -o %t/eh-frame-arm64
# RUN: llvm-objdump --macho --syms --indirect-symbols --unwind-info \
# RUN:   --dwarf=frames %t/eh-frame-arm64 | FileCheck %s -D#BASE=0x100000000 -D#DWARF_ENC=3
# RUN: llvm-nm -m %t/eh-frame-arm64 | FileCheck %s --check-prefix NO-EH-SYMS

# COM: ld -r %t/eh-frame-arm64.o -o %S/Inputs/eh-frame-arm64-r.o
# RUN: %lld -arch arm64 -lSystem -lc++ %S/Inputs/eh-frame-arm64-r.o -o %t/eh-frame-arm64-r
# RUN: llvm-objdump --macho --syms --indirect-symbols --unwind-info \
# RUN:   --dwarf=frames %t/eh-frame-arm64-r | FileCheck %s -D#BASE=0x100000000 -D#DWARF_ENC=3
# RUN: llvm-nm -m %t/eh-frame-arm64-r | FileCheck %s --check-prefix NO-EH-SYMS

# ALIGN:      Name: __eh_frame
# ALIGN-NEXT: Segment: __TEXT
# ALIGN-NEXT: Address:
# ALIGN-NEXT: Size:
# ALIGN-NEXT: Offset:
# ALIGN-NEXT: Alignment: [[#ALIGN]]

# NO-EH-SYMS-NOT: __eh_frame

# CHECK: Indirect symbols for (__DATA_CONST,__got) 2 entries
# CHECK: address                         index  name
# CHECK: 0x[[#%x,GXX_PERSONALITY_GOT:]]  {{.*}}  ___gxx_personality_v0
# CHECK: 0x[[#%x,MY_PERSONALITY_GOT:]]
# CHECK: SYMBOL TABLE:
# CHECK-DAG: [[#%x,F:]]              l   F __TEXT,__text _f
# CHECK-DAG: [[#%x,NO_UNWIND:]]      l   F __TEXT,__text _no_unwind
# CHECK-DAG: [[#%x,G:]]              l   F __TEXT,__text _g
# CHECK-DAG: [[#%x,H:]]              l   F __TEXT,__text _h
# CHECK-DAG: [[#%x,EXCEPT0:]]        l   O __TEXT,__gcc_except_tab GCC_except_table0
# CHECK-DAG: [[#%x,EXCEPT1:]]        l   O __TEXT,__gcc_except_tab GCC_except_table1
# CHECK-DAG: [[#%x,EXCEPT2:]]        l   O __TEXT,custom_except custom_except_table2
# CHECK-DAG: [[#%x,MY_PERSONALITY:]] g   F __TEXT,__text _my_personality
# CHECK: Contents of __unwind_info section:
# CHECK:   Version:                                   0x1
# CHECK:   Number of personality functions in array:  0x2
# CHECK:   Number of indices in array:                0x2
# CHECK:   Personality functions: (count = 2)
# CHECK:     personality[1]: 0x[[#%.8x,GXX_PERSONALITY_GOT - BASE]]
# CHECK:     personality[2]: 0x[[#%.8x,MY_PERSONALITY_GOT - BASE]]
# CHECK:   LSDA descriptors:
# CHECK:     [0]: function offset=0x[[#%.8x,F - BASE]], LSDA offset=0x[[#%.8x,EXCEPT0 - BASE]]
# CHECK:     [1]: function offset=0x[[#%.8x,G - BASE]], LSDA offset=0x[[#%.8x,EXCEPT1 - BASE]]
# CHECK:     [2]: function offset=0x[[#%.8x,H - BASE]], LSDA offset=0x[[#%.8x,EXCEPT2 - BASE]]
# CHECK:   Second level indices:
# CHECK:     Second level index[0]:
# CHECK:       [0]: function offset=0x[[#%.8x,F - BASE]],              encoding[{{.*}}]=0x52{{.*}}
# CHECK:       [1]: function offset=0x[[#%.8x,NO_UNWIND - BASE]],      encoding[{{.*}}]=0x00000000
# CHECK:       [2]: function offset=0x[[#%.8x,G - BASE]],              encoding[{{.*}}]=0x1[[#%x,DWARF_ENC]][[#%.6x, G_DWARF_OFF:]]
# CHECK:       [3]: function offset=0x[[#%.8x,H - BASE]],              encoding[{{.*}}]=0x2[[#%x,DWARF_ENC]][[#%.6x, H_DWARF_OFF:]]
# CHECK:       [4]: function offset=0x[[#%.8x,MY_PERSONALITY - BASE]], encoding[{{.*}}]=0x00000000

# CHECK: .debug_frame contents:
# CHECK: .eh_frame contents:

# CHECK: [[#%.8x,CIE1_OFF:]] {{.*}} CIE
# CHECK:   Format:                DWARF32
# CHECK:   Version:               1
# CHECK:   Augmentation:          "zPLR"
# CHECK:   Code alignment factor: 1
# CHECK:   Data alignment factor: -8
# CHECK:   Return address column:
# CHECK:   Personality Address:   [[#%.16x,GXX_PERSONALITY_GOT]]
# CHECK:   Augmentation data:     9B {{(([[:xdigit:]]{2} ){4})}}10 10

# CHECK: [[#%.8x,G_DWARF_OFF]] {{.*}} [[#%.8x,G_DWARF_OFF + 4 - CIE1_OFF]] FDE cie=[[#CIE1_OFF]] pc=[[#%x,G]]
# CHECK:   Format:       DWARF32
# CHECK:   LSDA Address: [[#%.16x,EXCEPT1]]
# CHECK:   DW_CFA_def_cfa_offset: +8
# CHECK:   0x[[#%x,G]]:

# CHECK: [[#%.8x,CIE2_OFF:]] {{.*}} CIE
# CHECK:   Format:                DWARF32
# CHECK:   Version:               1
# CHECK:   Augmentation:          "zPLR"
# CHECK:   Code alignment factor: 1
# CHECK:   Data alignment factor: -8
# CHECK:   Return address column:
# CHECK:   Personality Address:   [[#%.16x,MY_PERSONALITY_GOT]]
# CHECK:   Augmentation data:     9B {{(([[:xdigit:]]{2} ){4})}}10 10

# CHECK: [[#%.8x,H_DWARF_OFF]] {{.*}} [[#%.8x,H_DWARF_OFF + 4 - CIE2_OFF]] FDE cie=[[#CIE2_OFF]] pc=[[#%x,H]]
# CHECK:   Format:       DWARF32
# CHECK:   LSDA Address: [[#%.16x,EXCEPT2]]
# CHECK:   DW_CFA_def_cfa_offset: +8
# CHECK:   0x[[#%x,H]]:

.globl _my_personality, _main

.text
## _f's unwind info can be encoded with compact unwind, so we shouldn't see an
## FDE entry for it in the output file.
.p2align 2
_f:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception0
  .cfi_def_cfa_offset 8
  ret
  .cfi_endproc

.p2align 2
_no_unwind:
  ret

.p2align 2
_g:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception1
  .cfi_def_cfa_offset 8
  ## cfi_escape cannot be encoded in compact unwind, so we must keep _g's FDE
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

.p2align 2
_h:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  .cfi_lsda 16, Lexception2
  .cfi_def_cfa_offset 8
  ## cfi_escape cannot be encoded in compact unwind, so we must keep _h's FDE
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

.p2align 2
_my_personality:
  ret

.p2align 2
_main:
  ret

.section __TEXT,__gcc_except_tab
GCC_except_table0:
Lexception0:
  .byte 255

GCC_except_table1:
Lexception1:
  .byte 255

.section __TEXT,custom_except
custom_except_table2:
Lexception2:
  .byte 255

.subsections_via_symbols
