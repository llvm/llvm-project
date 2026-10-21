# REQUIRES: x86, aarch64
## Test that __eh_frame CIE/FDE ordering is preserved even when
## priority-based section sorting (from BP compression sort, order files,
## etc.) would otherwise reorder input sections. CIE records must precede
## the FDE records that reference them; reordering breaks CIE pointer
## resolution.

## x86_64
# RUN: llvm-mc -filetype=obj -emit-compact-unwind-non-canonical=true -triple=x86_64-apple-macos10.15 %s -o %t-x86_64.o
# RUN: %lld -lSystem -lc++ %t-x86_64.o -o %t-x86_64 --bp-compression-sort=both
# RUN: llvm-objdump --dwarf=frames %t-x86_64 2>&1 | FileCheck %s --implicit-check-not=error --implicit-check-not=warning

## arm64
# RUN: llvm-mc -filetype=obj -emit-compact-unwind-non-canonical=true -triple=arm64-apple-macos11.0 %s -o %t-arm64.o
# RUN: %lld -arch arm64 -lSystem -lc++ %t-arm64.o -o %t-arm64 --bp-compression-sort=both
# RUN: llvm-objdump --dwarf=frames %t-arm64 2>&1 | FileCheck %s --implicit-check-not=error --implicit-check-not=warning

## Verify that __eh_frame starts with a CIE (not an FDE), contains both
## CIEs and FDEs, and that no parse errors occur. The test uses two
## personality functions, producing two CIE groups (CIE_A + FDEs, CIE_B +
## FDEs). The --implicit-check-not flags above ensure no FDE fails to
## resolve its CIE pointer.
# CHECK: .eh_frame contents:
# CHECK: {{[0-9a-f]+}} {{.*}} CIE
# CHECK: {{[0-9a-f]+}} {{.*}} FDE
# CHECK: {{[0-9a-f]+}} {{.*}} FDE
# CHECK: {{[0-9a-f]+}} {{.*}} FDE
# CHECK: {{[0-9a-f]+}} {{.*}} FDE

.globl _my_personality_a, _my_personality_b, _main

.text
## _func_a uses cfi_escape to force DWARF unwind (can't be compact-encoded).
## Uses personality A -> produces CIE_A + FDE for _func_a.
.p2align 2
_func_a:
  .cfi_startproc
  .cfi_personality 155, _my_personality_a
  .cfi_lsda 16, Lexception_a
  .cfi_def_cfa_offset 16
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

## _func_b also uses personality A + cfi_escape -> reuses CIE_A, new FDE.
.p2align 2
_func_b:
  .cfi_startproc
  .cfi_personality 155, _my_personality_a
  .cfi_lsda 16, Lexception_b
  .cfi_def_cfa_offset 16
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

## _func_c uses personality B + cfi_escape -> produces CIE_B + FDE.
.p2align 2
_func_c:
  .cfi_startproc
  .cfi_personality 155, _my_personality_b
  .cfi_lsda 16, Lexception_c
  .cfi_def_cfa_offset 16
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

## _func_d uses personality B + cfi_escape -> reuses CIE_B, new FDE.
.p2align 2
_func_d:
  .cfi_startproc
  .cfi_personality 155, _my_personality_b
  .cfi_lsda 16, Lexception_d
  .cfi_def_cfa_offset 16
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

.p2align 2
_my_personality_a:
  ret

.p2align 2
_my_personality_b:
  ret

.p2align 2
_main:
  ret

.section __TEXT,__gcc_except_tab
GCC_except_table_a:
Lexception_a:
  .byte 255

GCC_except_table_b:
Lexception_b:
  .byte 255

GCC_except_table_c:
Lexception_c:
  .byte 255

GCC_except_table_d:
Lexception_d:
  .byte 255

.subsections_via_symbols
