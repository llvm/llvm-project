## Regression test for inlined_subroutine address-range corruption.
##
## BOLT updates DWARF lexical-scope ranges (DW_TAG_inlined_subroutine /
## lexical_block low_pc/high_pc and DW_AT_ranges) via
## BinaryFunction::translateInputToOutputRange(), which maps a boundary using
## its input offset relative to the start of the containing basic block:
##
##   OutAddr = BB.getOutputAddressRange().first + (InputOffset - BB.getOffset())
##
## This assumes intra-block byte offsets are preserved input->output. Any pass
## that changes instruction sizes within a block with scope boundary breaks that
## assumption. Here --plt=all rewrites three `call ext@PLT` (5 bytes, e8+rel32)
## into `call *ext@GOT(%rip)` (6 bytes, ff 15+rel32), growing the block by +3B
## before the inlined copy of inl() begins. As a result the inlined_subroutine's
## low_pc is emitted 3 bytes too early -- landing inside the preceding converted
## call instruction instead of on inl()'s first instruction (the `leaq`).
##
## The range stays within the parent, so `llvm-dwarfdump --verify` does not catch
## it; we check the exact low_pc against the disassembled instruction address.
##
## Source (compiled with -O2 -g -gdwarf-4):
##   extern long exta(long), extb(long), extc(long);   // resolved from a DSO -> PLT
##   static inline __attribute__((always_inline)) long inl(long x){ return x*3+7; }
##   __attribute__((noinline)) long foo(long a){
##     long t = exta(a); t = extb(t); t = extc(t); return inl(t);
##   }
##   int main(int argc, char **argv){ return (int)foo(argc); }
##
## The fix is enabled through --accurate-debug-ranges and should track all dwarf
## inline scopes, translating them accordingly.

# REQUIRES: system-linux

## Build a DSO so the exta/extb/extc calls go through the PLT.
# RUN: echo 'long exta(long x){return x+1;} long extb(long x){return x+2;} long extc(long x){return x+3;}' \
# RUN:   | %clang %cflags -fpic -shared -xc - -o %t.so
# RUN: %clang %cflags -gdwarf-4 %s %t.so -o %t.exe -Wl,-q -no-pie
# RUN: llvm-bolt %t.exe -o %t.bolt --plt=all --update-debug-sections
## Concatenate disassembly then debug-info so FileCheck can capture the
## instruction address and compare it against the inline scope's low_pc.
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt > %t.out
# RUN: llvm-dwarfdump --debug-info %t.bolt >> %t.out
# RUN: FileCheck %s < %t.out

## inl() is inlined into foo() right after the three (now GOT-indirect) calls.
## Its inlined_subroutine must start exactly at inl()'s first instruction.
# CHECK-LABEL: <foo>:
# CHECK: [[INL:[0-9a-f]+]]:{{.*}}leaq	(%rax,%rax,2), %rax
# CHECK: DW_TAG_inlined_subroutine
# CHECK: DW_AT_abstract_origin {{.*}}"inl"
# CHECK-NEXT: DW_AT_low_pc {{.*}}0x{{0*}}[[INL]])

	.att_syntax
	.file	"t.c"
	.text
	.globl	foo                             # -- Begin function foo
	.p2align	4, 0x90
	.type	foo,@function
foo:                                    # @foo
.Lfunc_begin0:
	.file	1 "." "t.c"
	.loc	1 9 0                           # t.c:9:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: foo:a <- $rdi
	pushq	%rax
	.cfi_def_cfa_offset 16
.Ltmp0:
	.loc	1 10 12 prologue_end            # t.c:10:12
	callq	exta@PLT
.Ltmp1:
	#DEBUG_VALUE: foo:a <- [DW_OP_LLVM_entry_value 1] $rdi
	#DEBUG_VALUE: foo:t <- $rax
	.loc	1 11 7                          # t.c:11:7
	movq	%rax, %rdi
	callq	extb@PLT
.Ltmp2:
	#DEBUG_VALUE: foo:t <- $rax
	.loc	1 12 7                          # t.c:12:7
	movq	%rax, %rdi
	callq	extc@PLT
.Ltmp3:
	#DEBUG_VALUE: foo:t <- $rax
	#DEBUG_VALUE: inl:x <- $rax
	.loc	1 6 16                          # t.c:6:16 @[ t.c:13:10 ]
	leaq	(%rax,%rax,2), %rax
.Ltmp4:
	addq	$7, %rax
.Ltmp5:
	.loc	1 13 3 epilogue_begin           # t.c:13:3
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Ltmp6:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	.loc	1 16 51 prologue_end            # t.c:16:51
	movslq	%edi, %rdi
.Ltmp7:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $edi
	.loc	1 16 47 is_stmt 0               # t.c:16:47
	jmp	foo                             # TAILCALL
.Ltmp8:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # DW_OP_reg5
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	0
	.quad	0
.Ldebug_loc2:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	80                              # DW_OP_reg0
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Lfunc_begin1-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.short	1                               # Loc expr size
	.byte	85                              # super-register DW_OP_reg5
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Lfunc_end1-.Lfunc_begin0
	.short	4                               # Loc expr size
	.byte	243                             # DW_OP_GNU_entry_value
	.byte	1                               # 1
	.byte	85                              # super-register DW_OP_reg5
	.byte	159                             # DW_OP_stack_value
	.quad	0
	.quad	0
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
	.byte	3                               # Abbreviation Code
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
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
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
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
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
	.byte	8                               # Abbreviation Code
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
	.byte	9                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	23                              # DW_FORM_sec_offset
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	11                              # Abbreviation Code
	.ascii	"\212\202\001"                  # DW_TAG_GNU_call_site_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.ascii	"\221B"                         # DW_AT_GNU_call_site_value
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	12                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	13                              # Abbreviation Code
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
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	14                              # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	15                              # Abbreviation Code
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
	.byte	16                              # Abbreviation Code
	.ascii	"\211\202\001"                  # DW_TAG_GNU_call_site
	.byte	1                               # DW_CHILDREN_yes
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.ascii	"\225B"                         # DW_AT_GNU_tail_call
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	17                              # Abbreviation Code
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
	.byte	1                               # Abbrev [1] 0xb:0x16c DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	29                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x7 DW_TAG_base_type
	.long	.Linfo_string3                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	3                               # Abbrev [3] 0x31:0x18 DW_TAG_subprogram
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	73                              # DW_AT_type
	.byte	1                               # DW_AT_inline
	.byte	4                               # Abbrev [4] 0x3d:0xb DW_TAG_formal_parameter
	.long	.Linfo_string6                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	73                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x49:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	8                               # DW_AT_byte_size
	.byte	5                               # Abbrev [5] 0x50:0x85 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	73                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x69:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0                    # DW_AT_location
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	73                              # DW_AT_type
	.byte	7                               # Abbrev [7] 0x78:0xf DW_TAG_variable
	.long	.Ldebug_loc1                    # DW_AT_location
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
	.long	73                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x87:0x1e DW_TAG_inlined_subroutine
	.long	49                              # DW_AT_abstract_origin
	.quad	.Ltmp3                          # DW_AT_low_pc
	.long	.Ltmp5-.Ltmp3                   # DW_AT_high_pc
	.byte	1                               # DW_AT_call_file
	.byte	13                              # DW_AT_call_line
	.byte	10                              # DW_AT_call_column
	.byte	9                               # Abbrev [9] 0x9b:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc2                    # DW_AT_location
	.long	61                              # DW_AT_abstract_origin
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0xa5:0x15 DW_TAG_GNU_call_site
	.long	213                             # DW_AT_abstract_origin
	.quad	.Ltmp1                          # DW_AT_low_pc
	.byte	11                              # Abbrev [11] 0xb2:0x7 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	3                               # DW_AT_GNU_call_site_value
	.byte	243
	.byte	1
	.byte	85
	.byte	0                               # End Of Children Mark
	.byte	12                              # Abbrev [12] 0xba:0xd DW_TAG_GNU_call_site
	.long	230                             # DW_AT_abstract_origin
	.quad	.Ltmp2                          # DW_AT_low_pc
	.byte	12                              # Abbrev [12] 0xc7:0xd DW_TAG_GNU_call_site
	.long	247                             # DW_AT_abstract_origin
	.quad	.Ltmp3                          # DW_AT_low_pc
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xd5:0x11 DW_TAG_subprogram
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	73                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xe0:0x5 DW_TAG_formal_parameter
	.long	73                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xe6:0x11 DW_TAG_subprogram
	.long	.Linfo_string8                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	73                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0xf1:0x5 DW_TAG_formal_parameter
	.long	73                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	13                              # Abbrev [13] 0xf7:0x11 DW_TAG_subprogram
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	73                              # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
	.byte	14                              # Abbrev [14] 0x102:0x5 DW_TAG_formal_parameter
	.long	73                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x108:0x5d DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	42                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x121:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc3                    # DW_AT_location
	.long	.Linfo_string14                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.long	42                              # DW_AT_type
	.byte	15                              # Abbrev [15] 0x130:0xd DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.long	.Linfo_string15                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	16                              # DW_AT_decl_line
	.long	357                             # DW_AT_type
	.byte	16                              # Abbrev [16] 0x13d:0x27 DW_TAG_GNU_call_site
	.long	80                              # DW_AT_abstract_origin
                                        # DW_AT_GNU_tail_call
	.quad	.Ltmp8                          # DW_AT_low_pc
	.byte	11                              # Abbrev [11] 0x14a:0x19 DW_TAG_GNU_call_site_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	21                              # DW_AT_GNU_call_site_value
	.byte	243
	.byte	1
	.byte	85
	.byte	16
	.ascii	"\377\377\377\377\017"
	.byte	26
	.byte	18
	.byte	16
	.byte	31
	.byte	37
	.byte	48
	.byte	32
	.byte	30
	.byte	16
	.byte	32
	.byte	36
	.byte	33
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	17                              # Abbrev [17] 0x165:0x5 DW_TAG_pointer_type
	.long	362                             # DW_AT_type
	.byte	17                              # Abbrev [17] 0x16a:0x5 DW_TAG_pointer_type
	.long	367                             # DW_AT_type
	.byte	2                               # Abbrev [2] 0x16f:0x7 DW_TAG_base_type
	.long	.Linfo_string16                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	".........clang version 23........................................................................................................................" # string offset=0 ;
.Linfo_string1:
	.asciz	"t.c"                           # string offset=146 ; t.c
.Linfo_string2:
	.asciz	"."                 # string offset=150 ; .
.Linfo_string3:
	.asciz	"int"                           # string offset=164 ; int
.Linfo_string4:
	.asciz	"inl"                           # string offset=168 ; inl
.Linfo_string5:
	.asciz	"long"                          # string offset=172 ; long
.Linfo_string6:
	.asciz	"x"                             # string offset=177 ; x
.Linfo_string7:
	.asciz	"exta"                          # string offset=179 ; exta
.Linfo_string8:
	.asciz	"extb"                          # string offset=184 ; extb
.Linfo_string9:
	.asciz	"extc"                          # string offset=189 ; extc
.Linfo_string10:
	.asciz	"foo"                           # string offset=194 ; foo
.Linfo_string11:
	.asciz	"main"                          # string offset=198 ; main
.Linfo_string12:
	.asciz	"a"                             # string offset=203 ; a
.Linfo_string13:
	.asciz	"t"                             # string offset=205 ; t
.Linfo_string14:
	.asciz	"argc"                          # string offset=207 ; argc
.Linfo_string15:
	.asciz	"argv"                          # string offset=212 ; argv
.Linfo_string16:
	.asciz	"char"                          # string offset=217 ; char
	.ident	"clang version 23"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
