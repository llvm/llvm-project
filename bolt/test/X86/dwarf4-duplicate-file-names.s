## Check that --update-debug-sections keeps duplicate entries of a DWARF 4
## line table file list at their input numbers. Line table rows and the
## DW_AT_decl_file and DW_AT_call_file values in .debug_info refer to files by
## number, so merging the duplicates attributed code and declarations to the
## wrong files.
##
## The input below is the output of
##   clang --target=x86_64-unknown-linux-gnu -O1 -g -gdwarf-4 \
##     -fdebug-compilation-dir=. -S main.c
## for these sources:
##
## a.h:
##   extern volatile int ga;
##   static inline __attribute__((always_inline)) void foo(int x) { ga = x; }
## c.h:
##   __attribute__((noinline)) int baz(int x) { return x * 3; }
## d.h:
##   extern volatile int gd;
##   static inline __attribute__((always_inline)) void qux(int x) { gd = x; }
## main.c:
##   #include "a.h"
##   #include "c.h"
##   #include "d.h"
##   volatile int ga, gd;
##   int main(int argc, char **argv) {
##     foo(argc);
##     qux(argc + 1);
##     return baz(argc);
##   }
##
## It was then edited to add file 3, a duplicate of ./c.h, and file 6, a
## duplicate of main.c. ./a.h and ./d.h moved to files 4 and 5. The second row
## of baz, the DW_AT_decl_file of baz, the last row of main, the
## DW_AT_decl_file of main and the DW_AT_call_file of the inlined qux refer to
## the duplicates.

# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=4 -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-line %t.exe > %t.exe.line
# RUN: llvm-dwarfdump --debug-line %t.bolt > %t.bolt.line
# RUN: FileCheck %s --check-prefix=FILES --input-file=%t.exe.line
# RUN: FileCheck %s --check-prefix=FILES --input-file=%t.bolt.line
# RUN: process-debug-line %t.bolt.line | FileCheck %s --check-prefix=ROWS
# RUN: llvm-dwarfdump --debug-info %t.exe | FileCheck %s --check-prefix=INFO
# RUN: llvm-dwarfdump --debug-info %t.bolt | FileCheck %s --check-prefix=INFO

## BOLT copies the input rows of a function that it does not emit.
# RUN: llvm-bolt %t.exe -o %t.skip.bolt --update-debug-sections \
# RUN:   --skip-funcs=baz
# RUN: llvm-dwarfdump --debug-line %t.skip.bolt > %t.skip.line
# RUN: FileCheck %s --check-prefix=FILES --input-file=%t.skip.line
# RUN: process-debug-line %t.skip.line | FileCheck %s --check-prefix=SKIP

# FILES:      include_directories[  1] = "."
# FILES-NEXT: file_names[  1]:
# FILES-NEXT:            name: "main.c"
# FILES-NEXT:       dir_index: 1
# FILES:      file_names[  2]:
# FILES-NEXT:            name: "./c.h"
# FILES-NEXT:       dir_index: 1
# FILES:      file_names[  3]:
# FILES-NEXT:            name: "./c.h"
# FILES-NEXT:       dir_index: 1
# FILES:      file_names[  4]:
# FILES-NEXT:            name: "./a.h"
# FILES-NEXT:       dir_index: 1
# FILES:      file_names[  5]:
# FILES-NEXT:            name: "./d.h"
# FILES-NEXT:       dir_index: 1
# FILES:      file_names[  6]:
# FILES-NEXT:            name: "main.c"
# FILES-NEXT:       dir_index: 1
# FILES-NOT:  file_names[

## Each row as: CU, line, column, the base name of its file, flags.
# ROWS:      main.c 1 53 c.h is_stmt prologue_end
# ROWS-NEXT: main.c 1 44 c.h{{$}}
# ROWS-NEXT: main.c 1 44 c.h end_sequence
# ROWS-NEXT: main.c 2 67 a.h is_stmt prologue_end
# ROWS-NEXT: main.c 7 12 main.c is_stmt{{$}}
# ROWS-NEXT: main.c 2 67 d.h is_stmt{{$}}
# ROWS-NEXT: main.c 8 10 main.c is_stmt{{$}}
# ROWS-NEXT: main.c 8 10 main.c is_stmt end_sequence
# ROWS-NOT:  {{.}}

# SKIP:      main.c 2 67 a.h is_stmt prologue_end
# SKIP-NEXT: main.c 7 12 main.c is_stmt{{$}}
# SKIP-NEXT: main.c 2 67 d.h is_stmt{{$}}
# SKIP-NEXT: main.c 8 10 main.c is_stmt{{$}}
# SKIP-NEXT: main.c 8 10 main.c is_stmt end_sequence
# SKIP-NEXT: main.c 1 53 c.h is_stmt prologue_end
# SKIP-NEXT: main.c 1 44 c.h{{$}}
# SKIP-NEXT: main.c 1 44 c.h end_sequence
# SKIP-NOT:  {{.}}

# INFO:      DW_AT_name ("baz")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/c.h")
# INFO:      DW_AT_name ("x")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/c.h")
# INFO:      DW_AT_name ("foo")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/a.h")
# INFO:      DW_AT_name ("x")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/a.h")
# INFO:      DW_AT_name ("qux")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/d.h")
# INFO:      DW_AT_name ("x")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/d.h")
# INFO:      DW_AT_name ("main")
# INFO-NEXT: DW_AT_decl_file ("{{.*}}/main.c")
# INFO:      DW_TAG_inlined_subroutine
# INFO-NEXT: DW_AT_abstract_origin ({{.*}} "foo")
# INFO-NOT:  DW_TAG
# INFO:      DW_AT_call_file ("{{.*}}/main.c")
# INFO:      DW_TAG_inlined_subroutine
# INFO-NEXT: DW_AT_abstract_origin ({{.*}} "qux")
# INFO-NOT:  DW_TAG
# INFO:      DW_AT_call_file ("{{.*}}/main.c")

	.file	"main.c"
	.file	1 "." "main.c"
	.text
	.globl	baz                             # -- Begin function baz
	.p2align	4
	.type	baz,@function
baz:                                    # @baz
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:
	#DEBUG_VALUE: baz:x <- $edi
                                        # kill: def $edi killed $edi def $rdi
	.file	2 "." "./c.h"
	.file	3 "." "./c.h"
	.loc	2 1 53 prologue_end             # ./c.h:1:53
	leal	(%rdi,%rdi,2), %eax
	.loc	3 1 44 is_stmt 0                # ./c.h:1:44
	retq
.Ltmp0:
.Lfunc_end0:
	.size	baz, .Lfunc_end0-baz
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: foo:x <- $edi
                                        # kill: def $edi killed $edi def $rdi
	.file	4 "." "./a.h"
	.loc	4 2 67 prologue_end is_stmt 1   # ./a.h:2:67 @[ main.c:6:3 ]
	movl	%edi, ga(%rip)
.Ltmp1:
	.loc	1 7 12                          # main.c:7:12
	leal	1(%rdi), %eax
.Ltmp2:
	#DEBUG_VALUE: qux:x <- $eax
	.file	5 "." "./d.h"
	.file	6 "." "main.c"
	.loc	5 2 67                          # ./d.h:2:67 @[ main.c:7:3 ]
	movl	%eax, gd(%rip)
.Ltmp3:
                                        # kill: def $edi killed $edi killed $rdi
	.loc	6 8 10                          # main.c:8:10
	jmp	baz                             # TAILCALL
.Ltmp4:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.type	ga,@object                      # @ga
	.bss
	.globl	ga
	.p2align	2, 0x0
ga:
	.long	0                               # 0x0
	.size	ga, 4

	.type	gd,@object                      # @gd
	.globl	gd
	.p2align	2, 0x0
gd:
	.long	0                               # 0x0
	.size	gd, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	53                              # DW_TAG_volatile_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\227B"                         # DW_AT_GNU_all_call_sites
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.ascii	"\225B"                         # DW_AT_GNU_tail_call
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x12f DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x15 DW_TAG_variable
	.long	.Linfo_string3                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	ga
	.byte	3                               # Abbrev [3] 0x3f:0x5 DW_TAG_volatile_type
	.long	68                              # DW_AT_type
	.byte	4                               # Abbrev [4] 0x44:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x4b:0x15 DW_TAG_variable
	.long	.Linfo_string5                  # DW_AT_name
	.long	63                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
	.byte	9                               # DW_AT_location
	.byte	3
	.quad	gd
	.byte	5                               # Abbrev [5] 0x60:0x27 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string9                  # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	68                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x79:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string7                  # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x87:0x14 DW_TAG_subprogram
	.long	.Linfo_string6                  # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.byte	1                               # DW_AT_inline
	.byte	8                               # Abbrev [8] 0x8f:0xb DW_TAG_formal_parameter
	.long	.Linfo_string7                  # DW_AT_name
	.byte	4                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	7                               # Abbrev [7] 0x9b:0x14 DW_TAG_subprogram
	.long	.Linfo_string8                  # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.byte	1                               # DW_AT_inline
	.byte	8                               # Abbrev [8] 0xa3:0xb DW_TAG_formal_parameter
	.long	.Linfo_string7                  # DW_AT_name
	.byte	5                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xaf:0x79 DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string10                 # DW_AT_name
	.byte	6                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	68                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0xc8:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	68                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0xd5:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	296                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0xe2:0x1c DW_TAG_inlined_subroutine
	.long	135                             # DW_AT_abstract_origin
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Ltmp1-.Lfunc_begin1            # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	6                               # DW_AT_call_line
	.byte	3                               # DW_AT_call_column
	.byte	10                              # Abbrev [10] 0xf6:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.long	143                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0xfe:0x1c DW_TAG_inlined_subroutine
	.long	155                             # DW_AT_abstract_origin
	.quad	.Ltmp2                          # DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   # DW_AT_high_pc
	.byte	6                               # DW_AT_call_file
	.byte	7                               # DW_AT_call_line
	.byte	3                               # DW_AT_call_column
	.byte	10                              # Abbrev [10] 0x112:0x7 DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	80
	.long	163                             # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	11                              # Abbrev [11] 0x11a:0xd DW_TAG_GNU_call_site
	.long	96                              # DW_AT_abstract_origin
                                        # DW_AT_GNU_tail_call
	.quad	.Ltmp4                          # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0x128:0x5 DW_TAG_pointer_type
	.long	301                             # DW_AT_type
	.byte	12                              # Abbrev [12] 0x12d:0x5 DW_TAG_pointer_type
	.long	306                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x132:0x7 DW_TAG_base_type
	.long	.Linfo_string13                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 21.1.0"
.Linfo_string1:
	.asciz	"main.c"
.Linfo_string2:
	.asciz	"."
.Linfo_string3:
	.asciz	"ga"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"gd"
.Linfo_string6:
	.asciz	"foo"
.Linfo_string7:
	.asciz	"x"
.Linfo_string8:
	.asciz	"qux"
.Linfo_string9:
	.asciz	"baz"
.Linfo_string10:
	.asciz	"main"
.Linfo_string11:
	.asciz	"argc"
.Linfo_string12:
	.asciz	"argv"
.Linfo_string13:
	.asciz	"char"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym ga
	.addrsig_sym gd
	.section	.debug_line,"",@progbits
.Lline_table_start0:
