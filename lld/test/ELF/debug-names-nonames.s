# The .s files were generated with:

# - clang++ -g -gpubnames -fdebug-compilation-dir='/proc/self/cwd' \
#    -S a.cpp -o a.names.s

# - clang++ -O0 -S a.cpp -o a.nonames.s

# - clang++ -O0 -S b.cpp -o b.s

# a.cpp contents:

# struct t1 { };
# void f1(t1) { }

# b.cpp contents:
# struct t1 { };
# int main() {
#   t1 v1;
# }
#

# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.names.s -o a.names.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.nonames.s -o a.nonames.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o

# Test one file with .debug_names and one file without.
# RUN: ld.lld --debug-names a.names.o b.o -o out0
# RUN: llvm-readelf -SW out0 | FileCheck %s --check-prefix=ELF1
# RUN: llvm-dwarfdump -debug-names out0 | FileCheck %s --check-prefix=DWARF

# DWARF:      .debug_names contents:
# DWARF-NEXT: Name Index @ 0x0 {
# DWARF-NEXT:   Header {
# DWARF-NEXT:     Length: 0x86
# DWARF-NEXT:     Format: DWARF32
# DWARF-NEXT:     Version: 5
# DWARF-NEXT:     CU count: 1
# DWARF-NEXT:     Local TU count: 0
# DWARF-NEXT:     Foreign TU count: 0
# DWARF-NEXT:     Bucket count: 3
# DWARF-NEXT:     Name count: 3
# DWARF-NEXT:     Abbreviations table size: 0x15
# DWARF-NEXT:     Augmentation: 'LLVM0700'
# DWARF-NEXT:   }
# DWARF-NEXT:   Compilation Unit offsets [
# DWARF-NEXT:     CU[0]: 0x00000000
# DWARF-NEXT:   ]
	
# ELF1: Name              Type     Address          Off      Size   ES Flg Lk Inf Al
# ELF1: .debug_names      PROGBITS 0000000000000000 [[#%x,]] 00008a 00      0   0  4

# Test both files without .debug_names.
# RUN: ld.lld --debug-names a.nonames.o b.o -o out
# RUN: llvm-readelf -SW out | FileCheck %s --check-prefix=ELF2

# Verify that there not a .debug_names section in the ELF file.
# ELF2: Name              Type     Address          Off      Size   ES Flg Lk Inf Al
# ELF2:  [ 1] .eh_frame         PROGBITS        0000000000200120 000120 00005c 00   A  0   0  8
# ELF2-NEXT:  [ 2] .text             PROGBITS        0000000000201180 000180 000018 00  AX  0   0 16
# ELF2-NEXT:  [ 3] .comment          PROGBITS        0000000000000000 000198 000071 01  MS  0   0  1
# ELF2-NEXT:  [ 4] .symtab           SYMTAB          0000000000000000 000210 000048 18      6   1  8
# ELF2-NEXT:  [ 5] .shstrtab         STRTAB          0000000000000000 000258 000034 00      0   0  1
# ELF2-NEXT:  [ 6] .strtab           STRTAB          0000000000000000 00028c 00000f 00      0   0  1
# ELF2-NEXT:Key to Flags:

#--- a.names.s

	.text
	.globl	_Z2f12t1                        # -- Begin function _Z2f12t1
	.p2align	4, 0x90
	.type	_Z2f12t1,@function
_Z2f12t1:                               # @_Z2f12t1
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f12t1, .Lfunc_end0-_Z2f12t1
	.cfi_endproc
                                        # -- End function
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.byte 0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git dfe787af70e11666a8e43ca73a86d1841d364dfc)" # string offset=0
.Linfo_string1:
	.asciz	"a.cpp"                         # string offset=104
.Linfo_string2:
	.asciz	"/proc/self/cwd"                # string offset=110
.Linfo_string3:
	.asciz	"f1"                            # string offset=125
.Linfo_string4:
	.asciz	"_Z2f12t1"                      # string offset=128
.Linfo_string5:
	.asciz	"t1"                            # string offset=137
	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
	.short	5                               # Header: version
	.short	0                               # Header: padding
	.long	1                               # Header: compilation unit count
	.long	0                               # Header: local type unit count
	.long	0                               # Header: foreign type unit count
	.long	3                               # Header: bucket count
	.long	3                               # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	8                               # Header: augmentation string size
	.ascii	"LLVM0700"                      # Header: augmentation string
	.long	.Lcu_begin0                     # Compilation unit 0
	.long	0                               # Bucket 0
	.long	1                               # Bucket 1
	.long	3                               # Bucket 2
	.long	5863324                         # Hash in Bucket 1
	.long	5863786                         # Hash in Bucket 1
	.long	1398125246                      # Hash in Bucket 2
	.long	.Linfo_string3                  # String in Bucket 1: f1
	.long	.Linfo_string5                  # String in Bucket 1: t1
	.long	.Linfo_string4                  # String in Bucket 2: _Z2f12t1
	.long	.Lnames0-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames2-.Lnames_entries0       # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0       # Offset in Bucket 2
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
.L1:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: f1
.Lnames2:
.L0:
	.byte	2                               # Abbreviation code
	.long	58                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: t1
.Lnames1:
	.byte	1                               # Abbreviation code
	.long	35                              # DW_IDX_die_offset
	.byte	0                               # DW_IDX_parent
                                        # End of list: _Z2f12t1
	.p2align	2, 0x0
.Lnames_end0:
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git dfe787af70e11666a8e43ca73a86d1841d364dfc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:

#--- a.nonames.s

	.text
	.globl	_Z2f12t1                        # -- Begin function _Z2f12t1
	.p2align	4, 0x90
	.type	_Z2f12t1,@function
_Z2f12t1:                               # @_Z2f12t1
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z2f12t1, .Lfunc_end0-_Z2f12t1
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git dfe787af70e11666a8e43ca73a86d1841d364dfc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

#--- b.s

	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 19.0.0git (git@github.com:llvm/llvm-project.git dfe787af70e11666a8e43ca73a86d1841d364dfc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
