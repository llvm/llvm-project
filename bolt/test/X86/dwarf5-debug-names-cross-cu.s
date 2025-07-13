
# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %tmain.o
# RUN: %clang %cflags -dwarf-5 %tmain.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --debug-info -r 0 --debug-names %t.bolt > %t.txt
# RUN: cat %t.txt | FileCheck --check-prefix=CHECK %s

## This test checks that BOLT generates Entries for DW_AT_abstract_origin when it has cross cu reference.

# CHECK: [[OFFSET1:0x[0-9a-f]*]]: Compile Unit
# CHECK: [[OFFSET2:0x[0-9a-f]*]]: Compile Unit
# CHECK:        Name Index @ 0x0 {
# CHECK-NEXT:     Header {
# CHECK-NEXT:       Length: 0xD2
# CHECK-NEXT:       Format: DWARF32
# CHECK-NEXT:       Version: 5
# CHECK-NEXT:       CU count: 2
# CHECK-NEXT:       Local TU count: 0
# CHECK-NEXT:       Foreign TU count: 0
# CHECK-NEXT:       Bucket count: 5
# CHECK-NEXT:       Name count: 5
# CHECK-NEXT:       Abbreviations table size: 0x1F
# CHECK-NEXT:       Augmentation: 'BOLT'
# CHECK-NEXT:     }
# CHECK-NEXT:     Compilation Unit offsets [
# CHECK-NEXT:       CU[0]: [[OFFSET1]]
# CHECK-NEXT:       CU[1]: [[OFFSET2]]
# CHECK-NEXT:     ]
# CHECK-NEXT:     Abbreviations [
# CHECK-NEXT:       Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
# CHECK-NEXT:         Tag: DW_TAG_subprogram
# CHECK-NEXT:         DW_IDX_compile_unit: DW_FORM_data1
# CHECK-NEXT:         DW_IDX_die_offset: DW_FORM_ref4
# CHECK-NEXT:         DW_IDX_parent: DW_FORM_flag_present
# CHECK-NEXT:       }
# CHECK-NEXT:       Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
# CHECK-NEXT:         Tag: DW_TAG_inlined_subroutine
# CHECK-NEXT:         DW_IDX_compile_unit: DW_FORM_data1
# CHECK-NEXT:         DW_IDX_die_offset: DW_FORM_ref4
# CHECK-NEXT:         DW_IDX_parent: DW_FORM_ref4
# CHECK-NEXT:       }
# CHECK-NEXT:       Abbreviation [[ABBREV3:0x[0-9a-f]*]] {
# CHECK-NEXT:         Tag: DW_TAG_base_type
# CHECK-NEXT:         DW_IDX_compile_unit: DW_FORM_data1
# CHECK-NEXT:         DW_IDX_die_offset: DW_FORM_ref4
# CHECK-NEXT:         DW_IDX_parent: DW_FORM_flag_present
# CHECK-NEXT:       }
# CHECK-NEXT:     ]
# CHECK-NEXT:     Bucket 0 [
# CHECK-NEXT:       EMPTY
# CHECK-NEXT:     ]
# CHECK-NEXT:     Bucket 1 [
# CHECK-NEXT:       Name 1 {
# CHECK-NEXT:         Hash: 0x7C9A7F6A
# CHECK-NEXT:         String: {{.+}} "main"
# CHECK-NEXT:         Entry @ [[ENTRY:0x[0-9a-f]*]] {
# CHECK-NEXT:           Abbrev: [[ABBREV1]]
# CHECK-NEXT:           Tag: DW_TAG_subprogram
# CHECK-NEXT:           DW_IDX_compile_unit: 0x00
# CHECK-NEXT:           DW_IDX_die_offset: 0x00000024
# CHECK-NEXT:           DW_IDX_parent: <parent not indexed>
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:       Name 2 {
# CHECK-NEXT:         Hash: 0xB5063CFE
# CHECK-NEXT:         String: {{.+}} "_Z3fooi"
# CHECK-NEXT:         Entry @ {{.+}} {
# CHECK-NEXT:           Abbrev: [[ABBREV1]]
# CHECK-NEXT:           Tag: DW_TAG_subprogram
# CHECK-NEXT:           DW_IDX_compile_unit: 0x01
# CHECK-NEXT:           DW_IDX_die_offset: 0x0000003a
# CHECK-NEXT:           DW_IDX_parent: <parent not indexed>
# CHECK-NEXT:         }
# CHECK-NEXT:         Entry @ {{.+}} {
# CHECK-NEXT:           Abbrev: [[ABBREV2]]
# CHECK-NEXT:           Tag: DW_TAG_inlined_subroutine
# CHECK-NEXT:           DW_IDX_compile_unit: 0x00
# CHECK-NEXT:           DW_IDX_die_offset: 0x00000054
# CHECK-NEXT:           DW_IDX_parent: Entry @ [[ENTRY]]
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:     ]
# CHECK-NEXT:     Bucket 2 [
# CHECK-NEXT:       EMPTY
# CHECK-NEXT:     ]
# CHECK-NEXT:     Bucket 3 [
# CHECK-NEXT:       Name 3 {
# CHECK-NEXT:         Hash: 0xB888030
# CHECK-NEXT:         String: {{.+}} "int"
# CHECK-NEXT:         Entry @ {{.+}} {
# CHECK-NEXT:           Abbrev: [[ABBREV3]]
# CHECK-NEXT:           Tag: DW_TAG_base_type
# CHECK-NEXT:           DW_IDX_compile_unit: 0x01
# CHECK-NEXT:           DW_IDX_die_offset: 0x00000036
# CHECK-NEXT:           DW_IDX_parent: <parent not indexed>
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:     ]
# CHECK-NEXT:     Bucket 4 [
# CHECK-NEXT:       Name 4 {
# CHECK-NEXT:         Hash: 0xB887389
# CHECK-NEXT:         String: {{.+}} "foo"
# CHECK-NEXT:         Entry @ {{.+}} {
# CHECK-NEXT:           Abbrev: [[ABBREV1]]
# CHECK-NEXT:           Tag: DW_TAG_subprogram
# CHECK-NEXT:           DW_IDX_compile_unit: 0x01
# CHECK-NEXT:           DW_IDX_die_offset: 0x0000003a
# CHECK-NEXT:           DW_IDX_parent: <parent not indexed>
# CHECK-NEXT:         }
# CHECK-NEXT:         Entry @ 0xc4 {
# CHECK-NEXT:           Abbrev: [[ABBREV2]]
# CHECK-NEXT:           Tag: DW_TAG_inlined_subroutine
# CHECK-NEXT:           DW_IDX_compile_unit: 0x00
# CHECK-NEXT:           DW_IDX_die_offset: 0x00000054
# CHECK-NEXT:           DW_IDX_parent: Entry @ [[ENTRY]]
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:       Name 5 {
# CHECK-NEXT:         Hash: 0x7C952063
# CHECK-NEXT:         String: {{.+}} "char"
# CHECK-NEXT:         Entry @ {{.+}} {
# CHECK-NEXT:           Abbrev: [[ABBREV3]]
# CHECK-NEXT:           Tag: DW_TAG_base_type
# CHECK-NEXT:           DW_IDX_compile_unit: 0x00
# CHECK-NEXT:           DW_IDX_die_offset: 0x00000075
# CHECK-NEXT:           DW_IDX_parent: <parent not indexed>
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:     ]
# CHECK-NEXT:   }

## clang++ -g2 -gpubnames -S -emit-llvm main.cpp -o main.ll
## clang++ -g2 -gpubnames -S -emit-llvm helper.cpp -o helper.ll
## llvm-link main.ll helper.ll -o combined.ll
## clang++ -g2 -gpubnames combined.ll -emit-llvm -S -o combined.opt.ll
## llc -dwarf-version=5 -filetype=asm -mtriple x86_64-unknown-linux combined.opt.ll -o combined.s
## main.cpp
## extern int foo(int);
## int main(int argc, char* argv[]) {
##   int i = 0;
##   [[clang::always_inline]] i = foo(argc);
##   return i;
## }
## helper.cpp
## int foo(int i) {
##   return i ++;
## }

	.text
	.file	"llvm-link"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/home" "main.cpp" md5 0x24fb0b4c3900e91fece1ac87ed73ff3b
	.loc	1 2 0                           # main.cpp:2:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$0, -16(%rbp)
	movl	%edi, -12(%rbp)
	movq	%rsi, -24(%rbp)
.Ltmp0:
	.loc	1 3 7 prologue_end              # main.cpp:3:7
	movl	$0, -4(%rbp)
	.loc	1 4 36                          # main.cpp:4:36
	movl	-12(%rbp), %eax
	movl	%eax, -8(%rbp)
.Ltmp1:
	.file	2 "/home" "helper.cpp" md5 0x7d4429e24d8c74d7ee22c1889ad46d6b
	.loc	2 2 12                          # helper.cpp:2:12
	movl	-8(%rbp), %eax
	movl	%eax, %ecx
	addl	$1, %ecx
	movl	%ecx, -8(%rbp)
.Ltmp2:
	.loc	1 4 30                          # main.cpp:4:30
	movl	%eax, -4(%rbp)
	.loc	1 5 10                          # main.cpp:5:10
	movl	-4(%rbp), %eax
	.loc	1 5 3 epilogue_begin is_stmt 0  # main.cpp:5:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.globl	_Z3fooi                         # -- Begin function _Z3fooi
	.p2align	4, 0x90
	.type	_Z3fooi,@function
_Z3fooi:                                # @_Z3fooi
.Lfunc_begin1:
	.loc	2 1 0 is_stmt 1                 # helper.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp4:
	.loc	2 2 12 prologue_end             # helper.cpp:2:12
	movl	-4(%rbp), %eax
	movl	%eax, %ecx
	addl	$1, %ecx
	movl	%ecx, -4(%rbp)
	.loc	2 2 3 epilogue_begin is_stmt 0  # helper.cpp:2:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp5:
.Lfunc_end1:
	.size	_Z3fooi, .Lfunc_end1-_Z3fooi
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	16                              # DW_FORM_ref_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	16                              # DW_FORM_ref_addr
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
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
	.byte	7                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	16                              # DW_FORM_ref_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x6d DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0x47 DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	.debug_info+174                 # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x32:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	116
	.byte	9                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	.debug_info+174                 # DW_AT_type
	.byte	4                               # Abbrev [4] 0x3d:0xb DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	104
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	106                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0x48:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	.debug_info+174                 # DW_AT_type
	.byte	6                               # Abbrev [6] 0x53:0x16 DW_TAG_inlined_subroutine
	.long	.debug_info+156                 # DW_AT_abstract_origin
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	4                               # DW_AT_call_line
	.byte	32                              # DW_AT_call_column
	.byte	7                               # Abbrev [7] 0x60:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.debug_info+165                 # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	8                               # Abbrev [8] 0x6a:0x5 DW_TAG_pointer_type
	.long	111                             # DW_AT_type
	.byte	8                               # Abbrev [8] 0x6f:0x5 DW_TAG_pointer_type
	.long	116                             # DW_AT_type
	.byte	9                               # Abbrev [9] 0x74:0x4 DW_TAG_base_type
	.byte	11                              # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x43 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	3                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	10                              # Abbrev [10] 0x23:0x12 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	53                              # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	11                              # Abbrev [11] 0x2c:0x8 DW_TAG_formal_parameter
	.byte	7                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	53                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	9                               # Abbrev [9] 0x35:0x4 DW_TAG_base_type
	.byte	6                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	12                              # Abbrev [12] 0x39:0x15 DW_TAG_subprogram
	.byte	2                               # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	35                              # DW_AT_abstract_origin
	.byte	13                              # Abbrev [13] 0x45:0x8 DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	44                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
	.section	.debug_str_offsets,"",@progbits
	.long	52                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git"       # string offset=0
.Linfo_string1:
	.asciz	"main.cpp"                      # string offset=24
.Linfo_string2:
	.asciz	"/home/ayermolo/local/tasks/T182867349" # string offset=33
.Linfo_string3:
	.asciz	"helper.cpp"                    # string offset=71
.Linfo_string4:
	.asciz	"_Z3fooi"                       # string offset=82
.Linfo_string5:
	.asciz	"foo"                           # string offset=90
.Linfo_string6:
	.asciz	"int"                           # string offset=94
.Linfo_string7:
	.asciz	"i"                             # string offset=98
.Linfo_string8:
	.asciz	"main"                          # string offset=100
.Linfo_string9:
	.asciz	"argc"                          # string offset=105
.Linfo_string10:
	.asciz	"argv"                          # string offset=110
.Linfo_string11:
	.asciz	"char"                          # string offset=115
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp1
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	2                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	5                               # Header: bucket count
	.long	5                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	.Lcu_begin1                     # Compilation unit 1
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	0                               # Bucket 2
	.long	3                               # Bucket 3
	.long	4                               # Bucket 4
	.long	2090499946                      # Hash in Bucket 1
	.long	-1257882370                     # Hash in Bucket 1
	.long	193495088                       # Hash in Bucket 3
	.long	193491849                       # Hash in Bucket 4
	.long	2090147939                      # Hash in Bucket 4
	.long	.Linfo_string8                  # String in Bucket 1: main
	.long	.Linfo_string4                  # String in Bucket 1: _Z3fooi
	.long	.Linfo_string6                  # String in Bucket 3: int
	.long	.Linfo_string5                  # String in Bucket 4: foo
	.long	.Linfo_string11                 # String in Bucket 4: char
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 4
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	1                               # DW_IDX_compile_unit
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
.L3:
	.byte	1                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames3:
.L0:
	.byte	1                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	57                              # DW_IDX_die_offset
.L2:                                    # DW_IDX_parent
	.byte	2                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	83                              # DW_IDX_die_offset
	.long	.L3-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _Z3fooi
.Lnames0:
.L4:
	.byte	3                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	53                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames2:
	.byte	1                               # Abbreviation code
	.byte	1                               # DW_IDX_compile_unit
	.long	57                              # DW_IDX_die_offset
	.byte	2                               # DW_IDX_parent
                                        # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	83                              # DW_IDX_die_offset
	.long	.L3-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: foo
.Lnames4:
.L1:
	.byte	3                               # Abbreviation code
	.byte	0                               # DW_IDX_compile_unit
	.long	116                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git"
	.ident	"clang version 19.0.0git"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
