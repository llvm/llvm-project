## Check that when BOLT inlines code from another CU, the file that it adds to
## the line table of the caller's CU gets the right directory and keeps its MD5.
## helper.s is assembled in inc with -fdebug-compilation-dir=./inc, so it is
## file #0 in directory #0, "./inc", of its DWARF 5 line table. BOLT inlines
## helper() into main(), whose CU is DWARF 5 in one run and DWARF 4 in the other.

# REQUIRES: system-linux

# RUN: rm -rf %t && split-file %s %t
# RUN: cd %t/inc && llvm-mc -g -dwarf-version=5 -fdebug-compilation-dir=./inc \
# RUN:   -filetype=obj -triple x86_64-unknown-linux helper.s -o %t/helper.o
# RUN: cd %t && llvm-mc -g -dwarf-version=5 -fdebug-compilation-dir=. \
# RUN:   -filetype=obj -triple x86_64-unknown-linux main.s -o %t/main5.o
# RUN: cd %t && llvm-mc -g -dwarf-version=4 -fdebug-compilation-dir=. \
# RUN:   -filetype=obj -triple x86_64-unknown-linux main.s -o %t/main4.o
# RUN: %clang %cflags %t/main5.o %t/helper.o -o %t/main5.exe -Wl,-q
# RUN: %clang %cflags %t/main4.o %t/helper.o -o %t/main4.exe -Wl,-q
# RUN: llvm-bolt %t/main5.exe -o %t/main5.bolt --update-debug-sections \
# RUN:   --force-inline=helper
# RUN: llvm-bolt %t/main4.exe -o %t/main4.bolt --update-debug-sections \
# RUN:   --force-inline=helper
# RUN: llvm-dwarfdump --debug-line %t/helper.o %t/main5.bolt \
# RUN:   | FileCheck %s --check-prefix=DW5
# RUN: llvm-dwarfdump --debug-line %t/main4.bolt \
# RUN:   | FileCheck %s --check-prefix=DW4

## The MD5 of helper.s in its own line table.
# DW5:      include_directories[  0] = "./inc"
# DW5-NEXT: file_names[  0]:
# DW5-NEXT:            name: "helper.s"
# DW5-NEXT:       dir_index: 0
# DW5-NEXT:    md5_checksum: [[MD5:[0-9a-f]+]]

## The DWARF 5 line table of main.s: helper.s is in "./inc" and keeps its MD5.
# DW5:      include_directories[  0] = "."
# DW5-NEXT: include_directories[  1] = "./inc"
# DW5-NEXT: file_names[  0]:
# DW5-NEXT:            name: "main.s"
# DW5-NEXT:       dir_index: 0
# DW5-NEXT:    md5_checksum: {{[0-9a-f]+}}
# DW5-NEXT: file_names[  1]:
# DW5-NEXT:            name: "helper.s"
# DW5-NEXT:       dir_index: 1
# DW5-NEXT:    md5_checksum: [[MD5]]
## The inlined instruction comes from line 5 of helper.s.
# DW5:      0x{{[0-9a-f]+}} 5 0 1 0 0 0 is_stmt

## The DWARF 4 line table of main.s has no MD5s, but helper.s is in "./inc".
# DW4:      include_directories[  1] = "./inc"
# DW4:      file_names[  2]:
# DW4-NEXT:            name: "helper.s"
# DW4-NEXT:       dir_index: 1
# DW4:      0x{{[0-9a-f]+}} 5 0 2 0 0 0 is_stmt

#--- main.s
  .text
  .globl main
  .type main,@function
main:
  callq helper
  retq
.Lmain_end:
  .size main, .Lmain_end-main

#--- inc/helper.s
  .text
  .globl helper
  .type helper,@function
helper:
	movl $0, %eax
  retq
.Lhelper_end:
  .size helper, .Lhelper_end-helper
