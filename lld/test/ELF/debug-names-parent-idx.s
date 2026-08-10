# debug-names-parent-idx.s generated with:

# clang++ -g -O0 -gpubnames -fdebug-compilation-dir='/proc/self/cwd' -S \
#    a.cpp -o a.s

# foo.h contents:

# int foo();

# struct foo {
#   int x;
#   char y;
#   struct foo *foo_ptr;
# };

# namespace parent_test {
#   int foo();
# }

#  a.cpp contents:

# #include "foo.h"
# void bar (struct foo &foo, int junk) {
#   foo.x = foo.x * junk;
# }
# int main (int argc, char** argv) {
#   struct foo my_struct;
#   my_struct.x = 10;
#   my_struct.y = 'q';
#   my_struct.foo_ptr = nullptr;
#   int junk = foo();
#   bar(my_struct, junk);
#   int junk2 = parent_test::foo();
#   return 0;
# }

# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld --debug-names a.o b.o -o out

# RUN: llvm-dwarfdump -debug-names out | FileCheck %s --check-prefix=DWARF

# DWARF:      .debug_names contents:
# DWARF:      Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length: 0x158
# DWARF-NEXT:     Format: DWARF32
# DWARF-NEXT:     Version: 5
# DWARF-NEXT:     CU count: 2
# DWARF-NEXT:     Local TU count: 0
# DWARF-NEXT:     Foreign TU count: 0
# DWARF-NEXT:     Bucket count: 9
# DWARF-NEXT:     Name count: 9
# DWARF-NEXT:     Abbreviations table size: 0x33
# DWARF-NEXT:     Augmentation: 'LLVM0700'
# DWARF:        Compilation Unit offsets [
# DWARF-NEXT:     CU[0]: 0x00000000
# DWARF-NEXT:     CU[1]: 0x0000000c
# DWARF:        Abbreviations [
# DWARF-NEXT:     Abbreviation 0x1 {
# DWARF-NEXT:       Tag: DW_TAG_base_type
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x2 {
# DWARF-NEXT:       Tag: DW_TAG_subprogram
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x3 {
# DWARF-NEXT:       Tag: DW_TAG_structure_type
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x4 {
# DWARF-NEXT:       Tag: DW_TAG_subprogram
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:          Abbreviation 0x5 {
# DWARF-NEXT:       Tag: DW_TAG_namespace
# DWARF-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# DWARF-NEXT:       DW_IDX_parent: DW_FORM_flag_present
# DWARF-NEXT:       DW_IDX_compile_unit: DW_FORM_data1
# DWARF:        Bucket 0 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 1 [
# DWARF-NEXT:     Name 1 {
# DWARF-NEXT:       Hash: 0xA974AA29
# DWARF-NEXT:       String: 0x00000174 "_ZN11parent_test3fooEv"
# DWARF-NEXT:       Entry @ 0x14a {
# DWARF-NEXT:         Abbrev: 0x4
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000045
# DWARF-NEXT:         DW_IDX_parent: Entry @ 0x128
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 2 {
# DWARF-NEXT:       Hash: 0xB5063D0B
# DWARF-NEXT:       String: 0x00000160 "_Z3foov"
# DWARF-NEXT:       Entry @ 0x155 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000027
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 2 [
# DWARF-NEXT:     Name 3 {
# DWARF-NEXT:       Hash: 0xB888030
# DWARF-NEXT:       String: 0x00000093 "int"
# DWARF-NEXT:       Entry @ 0xfe {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x0000008d
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0x104 {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 3 [
# DWARF-NEXT:     Name 4 {
# DWARF-NEXT:       Hash: 0xB8860BA
# DWARF-NEXT:       String: 0x0000007d "bar"
# DWARF-NEXT:       Entry @ 0xf7 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 5 {
# DWARF-NEXT:       Hash: 0xB887389
# DWARF-NEXT:       String: 0x00000097 "foo"
# DWARF-NEXT:       Entry @ 0x10b {
# DWARF-NEXT:         Abbrev: 0x3
# DWARF-NEXT:         Tag: DW_TAG_structure_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000096
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0x111 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000027
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0x117 {
# DWARF-NEXT:         Abbrev: 0x4
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000045
# DWARF-NEXT:         DW_IDX_parent: Entry @ 0x128
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0x121 {
# DWARF-NEXT:         Abbrev: 0x3
# DWARF-NEXT:         Tag: DW_TAG_structure_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000056
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 4 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 5 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 6 [
# DWARF-NEXT:     EMPTY
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 7 [
# DWARF-NEXT:     Name 6 {
# DWARF-NEXT:       Hash: 0x7C9A7F6A
# DWARF-NEXT:       String: 0x0000008e "main"
# DWARF-NEXT:       Entry @ 0x136 {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000046
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]
# DWARF-NEXT:   Bucket 8 [
# DWARF-NEXT:     Name 7 {
# DWARF-NEXT:       Hash: 0xA7255AE
# DWARF-NEXT:       String: 0x00000168 "parent_test"
# DWARF-NEXT:       Entry @ 0x128 {
# DWARF-NEXT:         Abbrev: 0x5
# DWARF-NEXT:         Tag: DW_TAG_namespace
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000043
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 8 {
# DWARF-NEXT:       Hash: 0x51007E98
# DWARF-NEXT:       String: 0x00000081 "_Z3barR3fooi"
# DWARF-NEXT:       Entry @ 0x12f {
# DWARF-NEXT:         Abbrev: 0x2
# DWARF-NEXT:         Tag: DW_TAG_subprogram
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000023
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:     Name 9 {
# DWARF-NEXT:       Hash: 0x7C952063
# DWARF-NEXT:       String: 0x0000009f "char"
# DWARF-NEXT:       Entry @ 0x13d {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x000000b8
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x00
# DWARF-NEXT:       }
# DWARF-NEXT:       Entry @ 0x143 {
# DWARF-NEXT:         Abbrev: 0x1
# DWARF-NEXT:         Tag: DW_TAG_base_type
# DWARF-NEXT:         DW_IDX_die_offset: 0x00000078
# DWARF-NEXT:         DW_IDX_parent: <parent not indexed>
# DWARF-NEXT:         DW_IDX_compile_unit: 0x01
# DWARF-NEXT:       }
# DWARF-NEXT:     }
# DWARF-NEXT:   ]

#--- a.s
	.text
	.globl	_Z3barR3fooi                    # -- Begin function _Z3barR3fooi
	.p2align	4, 0x90
	.type	_Z3barR3fooi,@function
_Z3barR3fooi:                           # @_Z3barR3fooi
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
.Ltmp0:
	movq	-8(%rbp), %rax
	movl	(%rax), %ecx
	imull	-12(%rbp), %ecx
	movq	-8(%rbp), %rax
	movl	%ecx, (%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z3barR3fooi, .Lfunc_end0-_Z3barR3fooi
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp2:
	movl	$10, -32(%rbp)
	movb	$113, -28(%rbp)
	movq	$0, -24(%rbp)
	callq	_Z3foov@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %esi
	leaq	-32(%rbp), %rdi
	callq	_Z3barR3fooi
	callq	_ZN11parent_test3fooEv@PLT
	movl	%eax, -40(%rbp)
	xorl	%eax, %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
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
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	72                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)" # string offset=0
.Linfo_string1:
	.asciz	"a.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"bar"                           # string offset=125
.Linfo_string4:
	.asciz	"_Z3barR3fooi"                  # string offset=129
.Linfo_string5:
	.asciz	"main"                          # string offset=142
.Linfo_string6:
	.asciz	"int"                           # string offset=147
.Linfo_string7:
	.asciz	"foo"                           # string offset=151
.Linfo_string8:
	.asciz	"x"                             # string offset=155
.Linfo_string9:
	.asciz	"y"                             # string offset=157
.Linfo_string10:
	.asciz	"char"                          # string offset=159
.Linfo_string11:
	.asciz	"foo_ptr"                       # string offset=164
.Linfo_string12:
	.asciz	"junk"                          # string offset=172
.Linfo_string13:
	.asciz	"argc"                          # string offset=177
.Linfo_string14:
	.asciz	"argv"                          # string offset=182
.Linfo_string15:
	.asciz	"my_struct"                     # string offset=187
.Linfo_string16:
	.asciz	"junk2"                         # string offset=197
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	6                               # Header: bucket count
	.long	6                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	1                               # Bucket 0
	.long	0                               # Bucket 1
	.long	2                               # Bucket 2
	.long	4                               # Bucket 3
	.long	5                               # Bucket 4
	.long	6                               # Bucket 5
	.long	193487034                       # Hash in Bucket 0
	.long	193495088                       # Hash in Bucket 2
	.long	1358986904                      # Hash in Bucket 2
	.long	193491849                       # Hash in Bucket 3
	.long	2090499946                      # Hash in Bucket 4
	.long	2090147939                      # Hash in Bucket 5
	.long	.Linfo_string3                  # String in Bucket 0: bar
	.long	.Linfo_string6                  # String in Bucket 2: int
	.long	.Linfo_string4                  # String in Bucket 2: _Z3barR3fooi
	.long	.Linfo_string7                  # String in Bucket 3: foo
	.long	.Linfo_string5                  # String in Bucket 4: main
	.long	.Linfo_string10                 # String in Bucket 5: char
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 0
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 4
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 5
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
.L2:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: bar
.Lnames3:
.L1:
	.byte	2                               # Abbreviation code
	.long	141                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames1:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z3barR3fooi
.Lnames4:
.L4:
	.byte	3                               # Abbreviation code
	.long	150                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
.Lnames2:
.L0:
	.byte	1                               # Abbreviation code
	.long	70                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: main
.Lnames5:
.L3:
	.byte	2                               # Abbreviation code
	.long	184                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z3barR3fooi
	.addrsig_sym _Z3foov
	.addrsig_sym _ZN11parent_test3fooEv
	.section	.debug_line,"",@progbits
.Lline_table_start0:
	
#--- b.s
# Generated with:

# clang++ -g -O0 -gpubnames -fdebug-compilation-dir='/proc/self/cwd' -S \
#     b.cpp -o b.s

# foo.h contents:

# int foo();

#struct foo {
#   int x;
#   char y;
#   struct foo *foo_ptr;
# };

# namespace parent_test {
#   int foo();
# }

# b.cpp contents:

# #include "foo.h"
# int foo () {
#   struct foo struct2;
#   struct2.x = 1024;
#   struct2.y = 'r';
#   struct2.foo_ptr = nullptr;
#   return struct2.x * (int) struct2.y;
# }

# namespace parent_test {
# int foo () {
#   return 25;
# }
# }

	.text
	.globl	_Z3foov                         # -- Begin function _Z3foov
	.p2align	4, 0x90
	.type	_Z3foov,@function
_Z3foov:                                # @_Z3foov
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	movl	$1024, -16(%rbp)                # imm = 0x400
	movb	$114, -12(%rbp)
	movq	$0, -8(%rbp)
	movl	-16(%rbp), %eax
	movsbl	-12(%rbp), %ecx
	imull	%ecx, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z3foov, .Lfunc_end0-_Z3foov
	.cfi_endproc
                                        # -- End function
	.globl	_ZN11parent_test3fooEv          # -- Begin function _ZN11parent_test3fooEv
	.p2align	4, 0x90
	.type	_ZN11parent_test3fooEv,@function
_ZN11parent_test3fooEv:                 # @_ZN11parent_test3fooEv
.Lfunc_begin1:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp2:
	movl	$25, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
.Lfunc_end1:
	.size	_ZN11parent_test3fooEv, .Lfunc_end1-_ZN11parent_test3fooEv
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
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
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	56                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)" # string offset=0
.Linfo_string1:
	.asciz	"b.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"int"                           # string offset=125
.Linfo_string4:
	.asciz	"foo"                           # string offset=129
.Linfo_string5:
	.asciz	"_Z3foov"                       # string offset=133
.Linfo_string6:
	.asciz	"parent_test"                   # string offset=141
.Linfo_string7:
	.asciz	"_ZN11parent_test3fooEv"        # string offset=153
.Linfo_string8:
	.asciz	"struct2"                       # string offset=176
.Linfo_string9:
	.asciz	"x"                             # string offset=184
.Linfo_string10:
	.asciz	"y"                             # string offset=186
.Linfo_string11:
	.asciz	"char"                          # string offset=188
.Linfo_string12:
	.asciz	"foo_ptr"                       # string offset=193
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
.Ldebug_addr_end0:
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	6                               # Header: bucket count
	.long	6                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	3                               # Bucket 2
	.long	5                               # Bucket 3
	.long	0                               # Bucket 4
	.long	6                               # Bucket 5
	.long	-1451972055                     # Hash in Bucket 1
	.long	-1257882357                     # Hash in Bucket 1
	.long	175265198                       # Hash in Bucket 2
	.long	193495088                       # Hash in Bucket 2
	.long	193491849                       # Hash in Bucket 3
	.long	2090147939                      # Hash in Bucket 5
	.long	.Linfo_string7                  # String in Bucket 1: _ZN11parent_test3fooEv
	.long	.Linfo_string5                  # String in Bucket 1: _Z3foov
	.long	.Linfo_string6                  # String in Bucket 2: parent_test
	.long	.Linfo_string3                  # String in Bucket 2: int
	.long	.Linfo_string4                  # String in Bucket 3: foo
	.long	.Linfo_string11                 # String in Bucket 5: char
	.long	.Lnames4-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames3-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 2
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 3
	.long	.Lnames5-.Lnames_entries0       # Offset in Bucket 5
.Lnames_abbrev_start0:
	.byte	1                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	2                               # Abbrev code
	.byte	46                              # DW_TAG_subprogram
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	3                               # Abbrev code
	.byte	57                              # DW_TAG_namespace
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	4                               # Abbrev code
	.byte	36                              # DW_TAG_base_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	5                               # Abbrev code
	.byte	19                              # DW_TAG_structure_type
	.byte	3                               # DW_IDX_die_offset
	.byte	19                              # DW_FORM_ref4
	.byte	4                               # DW_IDX_parent
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev
	.byte	0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames4:
.L3:
	.byte	1                               # Abbreviation code
	.long	69                              # DW_IDX_die_offset
	.long	.L5-.Lnames_entries0            # DW_IDX_parent
	.byte	0                               # End of list: _ZN11parent_test3fooEv
.Lnames2:
.L0:
	.byte	2                               # Abbreviation code
	.long	39                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z3foov
.Lnames3:
.L5:
	.byte	3                               # Abbreviation code
	.long	67                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: parent_test
.Lnames0:
.L2:
	.byte	4                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: int
.Lnames1:
	.byte	2                               # Abbreviation code
	.long	39                              # DW_IDX_die_offset
	.byte	1                               # DW_IDX_parent
                                        # Abbreviation code
	.long	69                              # DW_IDX_die_offset
	.long	.L5-.Lnames_entries0            # DW_IDX_parent
.L4:
	.byte	5                               # Abbreviation code
	.long	86                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: foo
.Lnames5:
.L1:
	.byte	4                               # Abbreviation code
	.long	120                             # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: char
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 53b14cd9ce2b57da73d173fc876d2e9e199f5640)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
