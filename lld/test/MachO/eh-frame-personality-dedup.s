# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/eh-frame.s -o %t/eh-frame.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/cu.s -o %t/cu.o
# RUN: %lld -dylib %t/cu.o %t/eh-frame.o -o %t/out

## Sanity check: we want our input to contain a section (and not symbol)
## relocation for the personality reference.
# RUN: llvm-readobj --relocations %t/cu.o | FileCheck %s --check-prefix=SECT-RELOC
# SECT-RELOC:      Section __compact_unwind {
# SECT-RELOC-NEXT:   __text
# SECT-RELOC-NEXT:   __text
# SECT-RELOC-NEXT: }

## Verify that the personality referenced via a symbol reloc in eh-frame.s gets
## dedup'ed with the personality referenced via a section reloc in cu.s.
# RUN: llvm-objdump --macho --unwind-info %t/out | FileCheck %s
# CHECK: Personality functions: (count = 1)

#--- eh-frame.s
_fun:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  ## cfi_escape cannot be encoded in compact unwind
  .cfi_escape 0
  ret
  .cfi_endproc

.subsections_via_symbols

#--- cu.s
.globl _my_personality
_fun:
  .cfi_startproc
  .cfi_personality 155, _my_personality
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

_my_personality:
  nop

.subsections_via_symbols
