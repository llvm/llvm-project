# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %s -o %t.o
# RUN: llvm-objdump --macho --reloc --section=__gcc_except_tab %t.o | \
# RUN:   FileCheck %s --check-prefix=INPUT
# RUN: %lld -dylib -arch arm64 -platform_version macos 11.0 11.0 \
# RUN:   -undefined dynamic_lookup --icf=all %t.o -o %t
# RUN: llvm-objdump --macho --syms --unwind-info --dwarf=frames %t | \
# RUN:   FileCheck %s

## Modelled on a real-world ObjC object containing identical methods that
## @catch NSException were not folded because their exception tables were not:
## the tables reference _OBJC_EHTYPE_$_NSException through PC-relative
## pointer-to-GOT relocations, so their raw bytes differ by the position of the
## relocation site. ICF must normalize the relocated bytes to fold the tables,
## after which the methods fold as well.
##
## _f0 and _f1 model those methods: identical bodies, compact unwind, and
## identical exception tables at different offsets. Both the tables and the
## functions must fold.
##
## _g0 and _g1 use an outlined prologue; the resulting CFI cannot be expressed
## as compact unwind, so they are unwound via DWARF FDEs. They share an
## exception table but deliberately have different bodies, so only the table is
## folded; _g1's FDE and its __unwind_info LSDA entry must then point at the
## surviving table, not at the folded one. DWARF unwinding is required to
## exercise this: the LSDA of a compact unwind entry is a relocation that gets
## canonicalized after ICF, whereas the LSDA of an FDE is resolved to an input
## section when the FDE is parsed and needs to be canonicalized separately.

## Sanity-check the input: each TType entry is a PC-relative pointer-to-GOT
## relocation, so the raw bytes of the four tables differ and they can only be
## folded after normalization. Without this the fold checks below could pass
## vacuously.
# INPUT:         Relocation information (__TEXT,__gcc_except_tab) 4 entries
# INPUT-COUNT-4: PTRTGOT False     _OBJC_EHTYPE_$_NSException

# CHECK-LABEL: SYMBOL TABLE:
# CHECK: [[#%x,LSDA0:]] l   O __TEXT,__gcc_except_tab GCC_except_table0
# CHECK: [[#LSDA0]]     l   O __TEXT,__gcc_except_tab GCC_except_table1
# CHECK: [[#%x,LSDA2:]] l   O __TEXT,__gcc_except_tab GCC_except_table2
# CHECK: [[#LSDA2]]     l   O __TEXT,__gcc_except_tab GCC_except_table3
# CHECK: [[#%x,F0:]]    g   F __TEXT,__text _f0
# CHECK: [[#F0]]        g   F __TEXT,__text _f1
# CHECK: [[#%x,G0:]]    g   F __TEXT,__text _g0
# CHECK: [[#%x,G1:]]    g   F __TEXT,__text _g1

# CHECK-LABEL: Contents of __unwind_info section:
# CHECK:      LSDA descriptors:
# CHECK-NEXT:   [0]: function offset=0x[[#%.8x,F0]], LSDA offset=0x[[#%.8x,LSDA0]]
# CHECK-NEXT:   [1]: function offset=0x[[#%.8x,G0]], LSDA offset=0x[[#%.8x,LSDA2]]
# CHECK-NEXT:   [2]: function offset=0x[[#%.8x,G1]], LSDA offset=0x[[#%.8x,LSDA2]]
# CHECK:      Second level indices:
# CHECK:        function offset=0x[[#%.8x,F0]], encoding[{{[0-9]+}}]=0x54000000
# CHECK-NEXT:   function offset=0x[[#%.8x,G0]], encoding[{{[0-9]+}}]=0x03{{[0-9a-f]+}}
# CHECK-NEXT:   function offset=0x[[#%.8x,G1]], encoding[{{[0-9]+}}]=0x03{{[0-9a-f]+}}

# CHECK-LABEL: .eh_frame contents:
# CHECK: FDE cie={{.+}} pc=[[#%.8x,G0]]...{{.+}}
# CHECK: LSDA Address: [[#%.16x,LSDA2]]
# CHECK: FDE cie={{.+}} pc=[[#%.8x,G1]]...{{.+}}
# CHECK: LSDA Address: [[#%.16x,LSDA2]]

.section __TEXT,__text,regular,pure_instructions
.globl _f0, _f1, _g0, _g1
.p2align 2

_f0:
Lfunc_begin0:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, Lexception0
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
Ltmp0:
  bl _callee0
Ltmp1:
  ldp x29, x30, [sp], #16
  ret
Ltmp2:
  bl _objc_begin_catch
  bl _objc_end_catch
  ldp x29, x30, [sp], #16
  ret
Lfunc_end0:
  .cfi_endproc

_f1:
Lfunc_begin1:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, Lexception1
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
Ltmp3:
  bl _callee0
Ltmp4:
  ldp x29, x30, [sp], #16
  ret
Ltmp5:
  bl _objc_begin_catch
  bl _objc_end_catch
  ldp x29, x30, [sp], #16
  ret
Lfunc_end1:
  .cfi_endproc

_g0:
Lfunc_begin2:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, Lexception2
  stp x29, x30, [sp, #-16]!
  bl _OUTLINED_FUNCTION_PROLOG_FRAME32_x30x29x19x20x21x22
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  .cfi_offset w19, -24
  .cfi_offset w20, -32
  .cfi_offset w21, -40
  .cfi_offset w22, -48
  .cfi_def_cfa w29, 16
Ltmp6:
  bl _callee1
Ltmp7:
  ldp x29, x30, [sp, #32]
  ldp x20, x19, [sp, #16]
  ldp x22, x21, [sp], #48
  ret
Ltmp8:
  bl _objc_begin_catch
  bl _objc_end_catch
  ldp x29, x30, [sp, #32]
  ldp x20, x19, [sp, #16]
  ldp x22, x21, [sp], #48
  ret
Lfunc_end2:
  .cfi_endproc

_g1:
Lfunc_begin3:
  .cfi_startproc
  .cfi_personality 155, ___objc_personality_v0
  .cfi_lsda 16, Lexception3
  stp x29, x30, [sp, #-16]!
  bl _OUTLINED_FUNCTION_PROLOG_FRAME32_x30x29x19x20x21x22
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  .cfi_offset w19, -24
  .cfi_offset w20, -32
  .cfi_offset w21, -40
  .cfi_offset w22, -48
  .cfi_def_cfa w29, 16
Ltmp9:
  bl _callee2
Ltmp10:
  ldp x29, x30, [sp, #32]
  ldp x20, x19, [sp, #16]
  ldp x22, x21, [sp], #48
  ret
Ltmp11:
  bl _objc_begin_catch
  bl _objc_end_catch
  ldp x29, x30, [sp, #32]
  ldp x20, x19, [sp, #16]
  ldp x22, x21, [sp], #48
  ret
Lfunc_end3:
  .cfi_endproc

.section __TEXT,__gcc_except_tab
.p2align 2
GCC_except_table0:
Lexception0:
  .byte 255                             ; @LPStart Encoding = omit
  .byte 155                             ; @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
  .byte 1                               ; Call site Encoding = uleb128
  .uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
  .uleb128 Lfunc_begin0-Lfunc_begin0    ; >> Call Site 1 <<
  .uleb128 Ltmp0-Lfunc_begin0           ;   Call between Lfunc_begin0 and Ltmp0
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
  .uleb128 Ltmp0-Lfunc_begin0           ; >> Call Site 2 <<
  .uleb128 Ltmp1-Ltmp0                  ;   Call between Ltmp0 and Ltmp1
  .uleb128 Ltmp2-Lfunc_begin0           ;     jumps to Ltmp2
  .byte 1                               ;   On action: 1
  .uleb128 Ltmp1-Lfunc_begin0           ; >> Call Site 3 <<
  .uleb128 Lfunc_end0-Ltmp1             ;   Call between Ltmp1 and Lfunc_end0
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
Lcst_end0:
  .byte 1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
  .byte 0                               ;   No further actions
  .p2align 2
                                        ; >> Catch TypeInfos <<
  .long _OBJC_EHTYPE_$_NSException@GOT-. ; TypeInfo 1
Lttbase0:
  .p2align 2

GCC_except_table1:
Lexception1:
  .byte 255                             ; @LPStart Encoding = omit
  .byte 155                             ; @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase1-Lttbaseref1
Lttbaseref1:
  .byte 1                               ; Call site Encoding = uleb128
  .uleb128 Lcst_end1-Lcst_begin1
Lcst_begin1:
  .uleb128 Lfunc_begin1-Lfunc_begin1    ; >> Call Site 1 <<
  .uleb128 Ltmp3-Lfunc_begin1           ;   Call between Lfunc_begin1 and Ltmp3
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
  .uleb128 Ltmp3-Lfunc_begin1           ; >> Call Site 2 <<
  .uleb128 Ltmp4-Ltmp3                  ;   Call between Ltmp3 and Ltmp4
  .uleb128 Ltmp5-Lfunc_begin1           ;     jumps to Ltmp5
  .byte 1                               ;   On action: 1
  .uleb128 Ltmp4-Lfunc_begin1           ; >> Call Site 3 <<
  .uleb128 Lfunc_end1-Ltmp4             ;   Call between Ltmp4 and Lfunc_end1
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
Lcst_end1:
  .byte 1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
  .byte 0                               ;   No further actions
  .p2align 2
                                        ; >> Catch TypeInfos <<
  .long _OBJC_EHTYPE_$_NSException@GOT-. ; TypeInfo 1
Lttbase1:
  .p2align 2

GCC_except_table2:
Lexception2:
  .byte 255                             ; @LPStart Encoding = omit
  .byte 155                             ; @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase2-Lttbaseref2
Lttbaseref2:
  .byte 1                               ; Call site Encoding = uleb128
  .uleb128 Lcst_end2-Lcst_begin2
Lcst_begin2:
  .uleb128 Lfunc_begin2-Lfunc_begin2    ; >> Call Site 1 <<
  .uleb128 Ltmp6-Lfunc_begin2           ;   Call between Lfunc_begin2 and Ltmp6
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
  .uleb128 Ltmp6-Lfunc_begin2           ; >> Call Site 2 <<
  .uleb128 Ltmp7-Ltmp6                  ;   Call between Ltmp6 and Ltmp7
  .uleb128 Ltmp8-Lfunc_begin2           ;     jumps to Ltmp8
  .byte 1                               ;   On action: 1
  .uleb128 Ltmp7-Lfunc_begin2           ; >> Call Site 3 <<
  .uleb128 Lfunc_end2-Ltmp7             ;   Call between Ltmp7 and Lfunc_end2
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
Lcst_end2:
  .byte 1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
  .byte 0                               ;   No further actions
  .p2align 2
                                        ; >> Catch TypeInfos <<
  .long _OBJC_EHTYPE_$_NSException@GOT-. ; TypeInfo 1
Lttbase2:
  .p2align 2

GCC_except_table3:
Lexception3:
  .byte 255                             ; @LPStart Encoding = omit
  .byte 155                             ; @TType Encoding = indirect pcrel sdata4
  .uleb128 Lttbase3-Lttbaseref3
Lttbaseref3:
  .byte 1                               ; Call site Encoding = uleb128
  .uleb128 Lcst_end3-Lcst_begin3
Lcst_begin3:
  .uleb128 Lfunc_begin3-Lfunc_begin3    ; >> Call Site 1 <<
  .uleb128 Ltmp9-Lfunc_begin3           ;   Call between Lfunc_begin3 and Ltmp9
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
  .uleb128 Ltmp9-Lfunc_begin3           ; >> Call Site 2 <<
  .uleb128 Ltmp10-Ltmp9                 ;   Call between Ltmp9 and Ltmp10
  .uleb128 Ltmp11-Lfunc_begin3          ;     jumps to Ltmp11
  .byte 1                               ;   On action: 1
  .uleb128 Ltmp10-Lfunc_begin3          ; >> Call Site 3 <<
  .uleb128 Lfunc_end3-Ltmp10            ;   Call between Ltmp10 and Lfunc_end3
  .byte 0                               ;     has no landing pad
  .byte 0                               ;   On action: cleanup
Lcst_end3:
  .byte 1                               ; >> Action Record 1 <<
                                        ;   Catch TypeInfo 1
  .byte 0                               ;   No further actions
  .p2align 2
                                        ; >> Catch TypeInfos <<
  .long _OBJC_EHTYPE_$_NSException@GOT-. ; TypeInfo 1
Lttbase3:
  .p2align 2

.subsections_via_symbols
