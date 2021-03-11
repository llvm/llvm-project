# REQUIRES: x86, shell
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-debug.s -o %t/no-debug.o
## Set modtimes of the files for deterministic test output.
# RUN: env TZ=UTC touch -t "197001010000.16" %t/test.o
# RUN: env TZ=UTC touch -t "197001010000.32" %t/foo.o
# RUN: llvm-ar rcsU %t/foo.a %t/foo.o

# RUN: %lld -lSystem %t/test.o %t/foo.o %t/no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; llvm-nm -pa %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o

## Check that we emit the right modtime even when the object file is in an
## archive.
# RUN: %lld -lSystem %t/test.o %t/foo.a %t/no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; llvm-nm -pa %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\)

## Check that we emit absolute paths to the object files in our OSO entries
## even if our inputs are relative paths.
# RUN: cd %t && %lld -lSystem test.o foo.o no-debug.o -o test
# RUN: (llvm-objdump --section-headers %t/test; llvm-nm -pa %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o

# RUN: cd %t && %lld -lSystem test.o foo.a no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; llvm-nm -pa %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\)

# CHECK:       Sections:
# CHECK-NEXT:  Idx                Name
# CHECK-NEXT:  [[#TEXT_ID:]]      __text
# CHECK-NEXT:  [[#DATA_ID:]]      __data
# CHECK-NEXT:  [[#MORE_DATA_ID:]] more_data
# CHECK-NEXT:  [[#COMM_ID:]]      __common
# CHECK-NEXT:  [[#MORE_TEXT_ID:]] more_text

# CHECK:       0000000000000000 - 00                     0000    SO /tmp/test.cpp
# CHECK-NEXT:  0000000000000010 - 03                     0001   OSO [[DIR]]/test.o
# CHECK-NEXT:  [[#%x, STATIC:]] - 0[[#MORE_DATA_ID + 1]] 0000 STSYM _static_var
# CHECK-NEXT:  [[#%x, MAIN:]]   - 0[[#TEXT_ID + 1]]      0000   FUN _main
# CHECK-NEXT:  0000000000000006 - 00                     0000   FUN
# CHECK-NEXT:  [[#%x, FUN:]]    - 0[[#MORE_TEXT_ID + 1]] 0000   FUN _fun
# CHECK-NEXT:  0000000000000001 - 00                     0000   FUN
# CHECK-NEXT:  [[#%x, GLOB:]]   - 0[[#DATA_ID + 1]]      0000  GSYM _global_var
# CHECK-NEXT:  [[#%x, ZERO:]]   - 0[[#COMM_ID + 1]]      0000  GSYM _zero
# CHECK-NEXT:  0000000000000000 - 01                     0000    SO
# CHECK-NEXT:  0000000000000000 - 00                     0000    SO /foo.cpp
# CHECK-NEXT:  0000000000000020 - 03                     0001   OSO [[FOO_PATH]]
# CHECK-NEXT:  [[#%x, FOO:]]    - 0[[#TEXT_ID + 1]]      0000   FUN _foo
# CHECK-NEXT:  0000000000000001 - 00                     0000   FUN
# CHECK-NEXT:  0000000000000000 - 01                     0000    SO
# CHECK-NEXT:  [[#STATIC]]      s _static_var
# CHECK-NEXT:  [[#MAIN]]        T _main
# CHECK-NEXT:  {{[0-9af]+}}     A _abs
# CHECK-NEXT:  [[#FUN]]         S _fun
# CHECK-NEXT:  [[#GLOB]]        D _global_var
# CHECK-NEXT:  [[#ZERO]]        S _zero
# CHECK-NEXT:  [[#FOO]]         T _foo
# CHECK-NEXT:  {{[0-9af]+}}     T _no_debug
# CHECK-EMPTY:

## Check that we don't attempt to emit rebase opcodes for the debug sections
## when building a PIE (since we have filtered the sections out).
# RUN: %lld -lSystem -pie %t/test.o %t/foo.a %t/no-debug.o -o %t/test
# RUN: llvm-objdump --macho --rebase %t/test | FileCheck %s --check-prefix=PIE
# PIE:       Rebase table:
# PIE-NEXT:  segment  section            address     type
# PIE-EMPTY:

#--- test.s

## Make sure we don't create STABS entries for absolute symbols.
.globl _abs
_abs = 0x123

.section __DATA, __data
.globl _global_var
_global_var:
  .quad 123

.section __DATA, more_data
_static_var:
  .quad 123

.globl  _zero
.zerofill __DATA,__common,_zero,4,2

.text
.globl  _main
_main:
Lfunc_begin0:
  callq _foo
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

.section OTHER,more_text,regular,pure_instructions
.globl _fun
_fun:
  ret

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

.section  __DWARF,__debug_aranges,regular,debug
ltmp1:
  .byte 0

#--- no-debug.s
## This file has no debug info.
.text
.globl _no_debug
_no_debug:
  ret
