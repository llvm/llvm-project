# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -dylib %t/test.o -o %t/test.dylib
# RUN: llvm-objdump --section-headers %t/test.dylib | FileCheck %s

## Test that lld can handle section-relative relocations in DWARF sections,
## specifically DW_FORM_strp which creates X86_64_RELOC_UNSIGNED relocations
## to the __debug_str section. This previously caused linker crashes with
## "malformed relocation" errors on macOS.
##
## The test verifies that:
## 1. The link completes successfully without crashing (key requirement)
## 2. DWARF sections are processed for relocations but not emitted to output
##    (MachO traditionally uses STABS, not DWARF, for debug info)
##
## Negative checks ensure DWARF sections are NOT in the final binary, preventing
## regression where they might be accidentally emitted.

# CHECK-NOT: __debug_info
# CHECK-NOT: __debug_abbrev
# CHECK-NOT: __debug_str

#--- test.s
.section __TEXT,__text,regular,pure_instructions
.globl _main
_main:
    movl $42, %eax
    retq

.section __DWARF,__debug_abbrev,regular,debug
Labbrev_begin:
    .byte 1                     ## Abbrev code
    .byte 17                    ## DW_TAG_compile_unit
    .byte 1                     ## DW_CHILDREN_yes
    .byte 37                    ## DW_AT_producer
    .byte 14                    ## DW_FORM_strp (string table pointer!)
    .byte 3                     ## DW_AT_name
    .byte 14                    ## DW_FORM_strp
    .byte 0                     ## End attributes
    .byte 0
    .byte 0                     ## End abbrev table

.section __DWARF,__debug_info,regular,debug
Linfo_begin:
    .long Linfo_end - Linfo_begin - 4  ## Length
    .short 4                    ## DWARF version 4
    .long 0                     ## Abbrev offset
    .byte 8                     ## Address size
    .byte 1                     ## Abbrev code
    ## These .long directives create section-relative relocations (X86_64_RELOC_UNSIGNED)
    ## to the __debug_str section. This is the critical test case.
    .long Lproducer - Ldebug_str  ## DW_AT_producer (section-relative!)
    .long Lfilename - Ldebug_str  ## DW_AT_name (section-relative!)
Linfo_end:

.section __DWARF,__debug_str,regular,debug
Ldebug_str:
Lproducer:
    .asciz "Test Producer 1.0"
Lfilename:
    .asciz "test.c"
