# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -dwarf-version=5 %s -o %t.o

# RUN: %lld -lSystem %t.o -o %t
# RUN: dsymutil -s %t | FileCheck %s -DDIR=%t -DSRC_PATH=%t.o

# CHECK:      (N_SO         ) 00      0000   0000000000000000   '/tmp{{[/\\]}}test.cpp'
# CHECK-NEXT: (N_OSO        ) 03      0001   {{.*}} '[[SRC_PATH]]'
# CHECK-NEXT: (N_FUN        ) 01      0000   [[#%.16x,MAIN:]]   '_main'
# CHECK-NEXT: (N_FUN        ) 00      0000   0000000000000001{{$}}
# CHECK-DAG:  (     SECT EXT) 01      0000   [[#MAIN]]           '_main'
# CHECK-DAG:  (       {{.*}}) {{[0-9]+}}                 0010   {{[0-9a-f]+}}      '__mh_execute_header'
# CHECK-DAG:  (       {{.*}}) {{[0-9]+}}                 0100   0000000000000000   'dyld_stub_binder'
# CHECK-EMPTY:

.text
.globl _main

.subsections_via_symbols

_main:
Lfunc_begin0:
  retq
Lfunc_end0:

.section  __DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
  .long   12                     ## Length of String Offsets Set
  .short  5
  .short  0
Lstr_offsets_base0:
.section  __DWARF,__debug_str,regular,debug
  .asciz  "test.cpp"             ## string offset=0
  .asciz  "/tmp"                 ## string offset=9
.section __DWARF,__debug_str_offs,regular,debug
  .long  0
  .long  9
.section  __DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
  .byte  1                       ## Abbreviation Code
  .byte  17                      ## DW_TAG_compile_unit
  .byte  1                       ## DW_CHILDREN_yes
  .byte  17                      ## DW_AT_low_pc
  .byte  1                       ## DW_FORM_addr
  .byte  18                      ## DW_AT_high_pc
  .byte  6                       ## DW_FORM_data4
  .byte  3                       ## DW_AT_name
  .byte  37                      ## DW_FORM_strx1
  .byte  27                      ## DW_AT_comp_dir
  .byte  37                      ## DW_FORM_strx1
  .byte  114                     ## DW_AT_str_offsets_base
  .byte  23                      ## DW_FORM_sec_offset
  .byte  0                       ## EOM(1)
  .byte  0                       ## EOM(2)
  .byte  0                       ## EOM(3)
.section  __DWARF,__debug_info,regular,debug
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
  .long  Lset0
Ldebug_info_start0:
  .short  5                       ## DWARF version number
  .byte   1                       ## DWARF Unit Type
  .byte   8                       ## Address Size (in bytes)
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
  .long  Lset1
  .byte  1                       ## Abbrev [1] 0xb:0x48 DW_TAG_compile_unit
  .quad  Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
  .long  Lset3
  .byte  0                       ## DW_AT_name
  .byte  1                       ## DW_AT_comp_dir
.set Lset4, Lstr_offsets_base0-Lsection_str_off  ## DW_AT_str_offsets_base
  .long Lset4
  .byte  0                       ## End Of Children Mark
Ldebug_info_end0:
