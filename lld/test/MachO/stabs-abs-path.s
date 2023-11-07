# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld -lSystem %t.o -o %t
# RUN: dsymutil -s %t | FileCheck %s

# CHECK:      (N_SO         ) 00      0000   0000000000000000   '/foo.cpp'

.text
.globl  _main
_main:
Lfunc_begin0:
  retq
Lfunc_end0:

.section  __DWARF,__debug_str,regular,debug
  .asciz  "/foo.cpp"             ## string offset=0
  .asciz  "/tmp"                 ## string offset=9
.section  __DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
  .byte  1                       ## Abbreviation Code
  .byte  17                      ## DW_TAG_compile_unit
  .byte  1                       ## DW_CHILDREN_yes
  .byte  3                       ## DW_AT_name
  .byte  14                      ## DW_FORM_strp
  .byte  27                      ## DW_AT_comp_dir
  .byte  14                      ## DW_FORM_strp
  .byte  17                      ## DW_AT_low_pc
  .byte  1                       ## DW_FORM_addr
  .byte  18                      ## DW_AT_high_pc
  .byte  6                       ## DW_FORM_data4
  .byte  0                       ## EOM(1)
  .byte  0                       ## EOM(2)
  .byte  0                       ## EOM(3)
.section  __DWARF,__debug_info,regular,debug
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
  .long  Lset0
Ldebug_info_start0:
  .short  4                       ## DWARF version number
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
  .long  Lset1
  .byte  8                       ## Address Size (in bytes)
  .byte  1                       ## Abbrev [1] 0xb:0x48 DW_TAG_compile_unit
  .long  0                       ## DW_AT_name
  .long  9                       ## DW_AT_comp_dir
  .quad  Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
  .long  Lset3
  .byte  0                       ## End Of Children Mark
Ldebug_info_end0:

.section  __DWARF,__debug_aranges,regular,debug
ltmp1:
  .byte 0

.subsections_via_symbols


