# REQUIRES: x86
# UNSUPPORTED: system-windows
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o

# RUN: %lld -lSystem %t/test.o %t/foo.o -o %t/test
# RUN: llvm-nm -pa %t/test | FileCheck %s -DDIR=%t

## Check that we emit absolute paths to the object files in our OSO entries
## even if our inputs are relative paths.
# RUN: cd %t && %lld -lSystem test.o foo.o -o test
# RUN: llvm-nm -pa %t/test | FileCheck %s -DDIR=%t

# CHECK-DAG:  [[#%x, MAIN:]]   T _main
# CHECK-DAG:  [[#%x, FOO: ]]   T _foo
# CHECK:      0000000000000000 - 00 0000    SO /tmp/test.cpp
# CHECK-NEXT: 0000000000000000 - 03 0001   OSO [[DIR]]/test.o
# CHECK-NEXT: [[#MAIN]]        - 01 0000   FUN _main
# CHECK-NEXT: 0000000000000001 - 00 0000   FUN
# CHECK-NEXT: 0000000000000000 - 01 0000    SO
# CHECK-NEXT: 0000000000000000 - 00 0000    SO /foo.cpp
# CHECK-NEXT: 0000000000000000 - 03 0001   OSO [[DIR]]/foo.o
# CHECK-NEXT: [[#FOO]]         - 01 0000   FUN _foo
# CHECK-NEXT: 0000000000000001 - 00 0000   FUN
# CHECK-NEXT: 0000000000000000 - 01 0000    SO

#--- test.s
.text
.globl  _main
_main:
Lfunc_begin0:
  retq
Lfunc_end0:

.section  __DWARF,__debug_str,regular,debug
  .asciz  "test.cpp"             ## string offset=0
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
.subsections_via_symbols
.section  __DWARF,__debug_line,regular,debug

#--- foo.s
.text
.globl  _foo
_foo:
Lfunc_begin0:
  retq
Lfunc_end0:

.section  __DWARF,__debug_str,regular,debug
  .asciz  "foo.cpp"              ## string offset=0
  .asciz  ""                     ## string offset=8
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
  .long  8                       ## DW_AT_comp_dir
  .quad  Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
  .long  Lset3
  .byte  0                       ## End Of Children Mark
Ldebug_info_end0:
.subsections_via_symbols
.section  __DWARF,__debug_line,regular,debug
