# REQUIRES: x86, shell
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-debug.s -o %t/no-debug.o
## Set modtimes of the files for deterministic test output.
# RUN: env TZ=GMT touch -t "197001010000.16" %t/test.o
# RUN: env TZ=GMT touch -t "197001010000.32" %t/foo.o
# RUN: llvm-ar rcsU %t/foo.a %t/foo.o

# RUN: ZERO_AR_DATE=0 %lld -lSystem %t/test.o %t/foo.o %t/no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o \
# RUN:       -D#TEST_TIME=0x10 -D#FOO_TIME=0x20

## Check that we emit the right modtime even when the object file is in an
## archive.
# RUN: ZERO_AR_DATE=0 %lld -lSystem %t/test.o %t/foo.a %t/no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\) \
# RUN:       -D#TEST_TIME=0x10 -D#FOO_TIME=0x20

## Check that we don't emit modtimes if ZERO_AR_DATE is set.
# RUN: env ZERO_AR_DATE=1 %lld -lSystem %t/test.o %t/foo.o %t/no-debug.o \
# RUN:     -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o \
# RUN:       -D#TEST_TIME=0 -D#FOO_TIME=0
# RUN: env %lld -lSystem %t/test.o %t/foo.a %t/no-debug.o \
# RUN:     -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\) \
# RUN:       -D#TEST_TIME=0 -D#FOO_TIME=0
# RUN: env %lld -lSystem %t/test.o %t/no-debug.o \
# RUN:     -all_load %t/foo.a -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\) \
# RUN:       -D#TEST_TIME=0 -D#FOO_TIME=0
# RUN: env %lld -lSystem %t/test.o %t/no-debug.o \
# RUN:     -force_load %t/foo.a -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\) \
# RUN:       -D#TEST_TIME=0 -D#FOO_TIME=0
# RUN: env ZERO_AR_DATE=0 %lld -lSystem -reproducible %t/test.o %t/foo.o \
# RUN:     %t/no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o \
# RUN:       -D#TEST_TIME=0 -D#FOO_TIME=0

## Check that we emit absolute paths to the object files in our OSO entries
## even if our inputs are relative paths.
# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem test.o foo.o no-debug.o -o test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.o \
# RUN:       -D#TEST_TIME=0x10 -D#FOO_TIME=0x20

## Check that we emit relative path to object files in OSO entries
## when -oso_prefix <path> is used.
# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem test.o foo.o no-debug.o -oso_prefix "%t" -o %t/test-rel
# RUN: dsymutil -s  %t/test-rel | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-PATH
# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem test.o foo.o no-debug.o -oso_prefix "%t/" -o %t/test-rel
# RUN: dsymutil -s  %t/test-rel | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-PATH-NO-SLASH
# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem test.o foo.o no-debug.o -oso_prefix "." -o %t/test-rel-dot
# RUN: dsymutil -s  %t/test-rel-dot | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-DOT
# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem ./test.o ./foo.o ./no-debug.o -oso_prefix "." -o %t/test-rel-dot
# RUN: dsymutil -s  %t/test-rel-dot | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-DOT-EXPLICIT

## Check that symlinks are not expanded when -oso_prefix . is used.
# RUN: mkdir -p %t/private/var/folders/tmp && ln -s private/var %t/var
# RUN: cp %t/test.o %t/foo.o %t/no-debug.o %t/private/var/folders/tmp
# RUN: env TZ=GMT touch -t "197001010000.16" %t/private/var/folders/tmp/test.o
# RUN: env TZ=GMT touch -t "197001010000.32" %t/private/var/folders/tmp/foo.o
# RUN: cd %t/var/folders/tmp && ZERO_AR_DATE=0 %lld -lSystem test.o foo.o no-debug.o -oso_prefix "." -o test-rel-symlink
# RUN: dsymutil -s  %t/private/var/folders/tmp/test-rel-symlink | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-DOT
# RUN: cd %t/var/folders/tmp && ZERO_AR_DATE=0 %lld -lSystem ./test.o ./foo.o ./no-debug.o -oso_prefix "." -o test-rel-symlink
# RUN: dsymutil -s  %t/private/var/folders/tmp/test-rel-symlink | grep 'N_OSO' | FileCheck %s  -D#TEST_TIME=0x10 -D#FOO_TIME=0x20 --check-prefix=REL-DOT-EXPLICIT

## Check that we don't emit DWARF or stabs when -S is used
# RUN: %lld -lSystem test.o foo.o no-debug.o -S -o %t/test-no-debug
## grep returns an exit code of 1 if it cannot match the intended pattern. We
## expect to not find any entries which requires the exit code to be negated.
# RUN: llvm-nm -ap %t/test-no-debug | not grep -e ' - '

# RUN: cd %t && ZERO_AR_DATE=0 %lld -lSystem test.o foo.a no-debug.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | \
# RUN:   FileCheck %s -DDIR=%t -DFOO_PATH=%t/foo.a\(foo.o\) \
# RUN:       -D#TEST_TIME=0x10 -D#FOO_TIME=0x20

# CHECK:       Sections:
# CHECK-NEXT:  Idx                Name
# CHECK-NEXT:  [[#TEXT_ID:]]      __text
# CHECK-NEXT:  [[#DATA_ID:]]      __data
# CHECK-NEXT:  [[#MORE_DATA_ID:]] more_data
# CHECK-NEXT:  [[#COMM_ID:]]      __common
# CHECK-NEXT:  [[#MORE_TEXT_ID:]] more_text

# CHECK:      (N_SO         ) 00                         0000   0000000000000000   '/tmp/test.cpp'
# CHECK-NEXT: (N_OSO        ) 03                         0001   [[#%.16x,TEST_TIME]] '[[DIR]]/test.o'
# REL-PATH:   (N_OSO        ) 03                         0001   [[#%.16x,TEST_TIME]] '/test.o'
# REL-PATH-NO-SLASH:  (N_OSO        ) 03                 0001   [[#%.16x,TEST_TIME]] 'test.o'
# REL-DOT:    (N_OSO        ) 03                         0001   [[#%.16x,TEST_TIME]] 'test.o'
# REL-DOT-EXPLICIT:    (N_OSO        ) 03                0001   [[#%.16x,TEST_TIME]] './test.o'
# CHECK-NEXT: (N_STSYM      ) [[#%.2d,MORE_DATA_ID + 1]] 0000   [[#%.16x,STATIC:]] '_static_var'
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,MAIN:]]   '_main'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000006{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,BAR:]]    '_bar'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000000{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,BAR2:]]   '_bar2'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000001{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,BAZ:]]    '_baz'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000000{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,BAZ2:]]   '_baz2'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000002{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,QUX:]]    '_qux'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000003{{$}}
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,QUUX:]]   '_quux'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000004{{$}}
# CHECK-NEXT: (N_GSYM       ) [[#%.2d,DATA_ID + 1]]      0000   [[#%.16x,GLOB:]]   '_global_var'
# CHECK-NEXT: (N_GSYM       ) [[#%.2d,COMM_ID + 1]]      0000   [[#%.16x,ZERO:]]   '_zero'
# CHECK-NEXT: (N_FUN        ) [[#%.2d,MORE_TEXT_ID + 1]] 0000   [[#%.16x,FUN:]]    '_fun'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000001{{$}}
# CHECK-NEXT: (N_SO         ) 01                         0000   0000000000000000{{$}}
# CHECK-NEXT: (N_SO         ) 00                         0000   0000000000000000   '/foo.cpp'
# CHECK-NEXT: (N_OSO        ) 03                         0001   [[#%.16x,FOO_TIME]] '[[FOO_PATH]]'
# REL-PATH-NEXT:   (N_OSO        ) 03                    0001   [[#%.16x,FOO_TIME]] '/foo.o'
# REL-PATH-NO-SLASH-NEXT:   (N_OSO        ) 03           0001   [[#%.16x,FOO_TIME]] 'foo.o'
# REL-DOT-NEXT:    (N_OSO        ) 03                    0001   [[#%.16x,FOO_TIME]] 'foo.o'
# CHECK-NEXT: (N_FUN        ) [[#%.2d,TEXT_ID + 1]]      0000   [[#%.16x,FOO:]]    '_foo'
# CHECK-NEXT: (N_FUN        ) 00                         0000   0000000000000001{{$}}
# CHECK-NEXT: (N_SO         ) 01                         0000   0000000000000000{{$}}
# CHECK-DAG:  (     SECT    ) [[#%.2d,MORE_DATA_ID + 1]] 0000   [[#STATIC]]        '_static_var'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#MAIN]]          '_main'
# CHECK-DAG:  (     ABS  EXT) 00                         0000   {{[0-9af]+}}       '_abs'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#FOO]]           '_foo'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#BAR]]           '_bar'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#BAR2]]          '_bar2'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#BAZ]]           '_baz'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#BAZ2]]          '_baz2'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#QUX]]           '_qux'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   [[#QUUX]]          '_quux'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,DATA_ID + 1]]      0000   [[#GLOB]]          '_global_var'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,COMM_ID + 1]]      0000   [[#ZERO]]          '_zero'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,MORE_TEXT_ID + 1]] 0000   [[#FUN]]           '_fun'
# CHECK-DAG:  (     SECT EXT) [[#%.2d,TEXT_ID + 1]]      0000   {{[0-9a-f]+}}      '_no_debug'
# CHECK-DAG:  (       {{.*}}) {{[0-9]+}}                 0010   {{[0-9a-f]+}}      '__mh_execute_header'
# CHECK-DAG:  (       {{.*}}) {{[0-9]+}}                 0100   0000000000000000   'dyld_stub_binder'
# CHECK-EMPTY:

## Check that we don't attempt to emit rebase opcodes for the debug sections
## when building a PIE (since we have filtered the sections out).
# RUN: %lld -lSystem %t/test.o %t/foo.a %t/no-debug.o -o %t/test
# RUN: llvm-objdump --macho --rebase %t/test | FileCheck %s --check-prefix=PIE
# PIE:       Rebase table:
# PIE-NEXT:  segment  section            address     type
# PIE-EMPTY:

## Check that an absolute DW_AT_name does not have DW_AT_comp_dir prepended
## when forming N_SO.
# RUN: llvm-mc -filetype obj -triple=x86_64-apple-darwin %t/abs-path.s -o %t/abs-path.o
# RUN: %lld %t/abs-path.o -o %t/test
# RUN: (llvm-objdump --section-headers %t/test; dsymutil -s %t/test) | FileCheck %s --check-prefix=ABS-PATH
# ABS-PATH:      (N_SO         ) 00      0000   0000000000000000   '/foo.cpp'

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
.globl  _main, _bar, _bar2, _baz, _baz2, _qux, _quux
.alt_entry _baz
.alt_entry _qux

_bar:
_bar2:
  .space 1

_baz:
_baz2:
  .space 2

_main:
Lfunc_begin0:
  callq _foo
  retq
Lfunc_end0:

_qux:
  .space 3

_quux:
  .space 4

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

.section OTHER,more_text,regular,pure_instructions
.globl _fun
_fun:
  ret

.subsections_via_symbols

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
  .long  8                       ## DW_AT_comp_dir
  .quad  Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
  .long  Lset3
  .byte  0                       ## End Of Children Mark
Ldebug_info_end0:

.section  __DWARF,__debug_aranges,regular,debug
ltmp1:
  .byte 0

.subsections_via_symbols

#--- no-debug.s
## This file has no debug info.
.text
.globl _no_debug
_no_debug:
  ret

#--- abs-path.s
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
