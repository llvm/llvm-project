# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-addr %t.bolt > %t.txt
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt >> %t.txt
# RUN: cat %t.txt | FileCheck --check-prefix=POSTCHECK %s

# This tests conversion for DWARF5 ranges DW_AT_ranges [DW_FORM_sec_offset] to DW_AT_ranges [DW_FORM_rnglistx]

# PRECHECK: version = 0x0005
# PRECHECK: DW_AT_ranges [DW_FORM_sec_offset]

# POSTCHECK: Addrs: [
# POSTCHECK-NEXT: 0x[[#%.16x,ADDR:]]
# POSTCHECK-NEXT: 0x[[#%.16x,ADDR1:]]
# POSTCHECK: version = 0x0005
# POSTCHECK: DW_AT_ranges [DW_FORM_rnglistx]
# POSTCHECK-SAME: (indexed (0x0) rangelist = 0x00000018
# POSTCHECK-NEXT: [0x[[#ADDR]], 0x[[#ADDR + 11]]
# POSTCHECK-NEXT: [0x[[#ADDR1]], 0x[[#ADDR1 + 11]]
# POSTCHECK-NEXT:  DW_AT_low_pc [DW_FORM_addr]
# POSTCHECK-NEXT:  DW_AT_stmt_list [DW_FORM_sec_offset]
# POSTCHECK-NEXT:  DW_AT_rnglists_base [DW_FORM_sec_offset]  (0x0000000c)
# POSTCHECK-NEXT:  DW_AT_addr_base [DW_FORM_sec_offset]  (0x00000008)


# int foo() {
#   return 3;
# }
#
# int main() {
#   return foo();
# }

	.file	"main.cpp"
	.text
.Ltext0:
	.file 0 "/testRangeOffsetToRangeX" "main.cpp"
	.section	.text._Z3foov,"ax",@progbits
	.globl	_Z3foov
	.type	_Z3foov, @function
_Z3foov:
.LFB0:
	.file 1 "main.cpp"
	.loc 0 1 11
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	.loc 0 2 10
	movl	$3, %eax
	.loc 0 3 1
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	_Z3foov, .-_Z3foov
	.section	.text.main,"ax",@progbits
	.globl	main
	.type	main, @function
main:
.LFB1:
	.loc 0 5 12
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	.loc 0 6 13
	call	_Z3foov
	.loc 0 6 14
	nop
	.loc 0 7 1
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.text
.Letext0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x6e
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x1
	.long	.LASF2
	.byte	0x21
	.long	.LASF0
	.long	.LASF1
	.long	.LLRL0
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x2
	.long	.LASF3
	.byte	0x1
	.byte	0x5
	.byte	0x5
	.long	0x48
	.quad	.LFB1
	.quad	.LFE1-.LFB1
	.uleb128 0x1
	.byte	0x9c
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x4
	.string	"foo"
	.byte	0x1
	.byte	0x1
	.byte	0x5
	.long	.LASF4
	.long	0x48
	.quad	.LFB0
	.quad	.LFE0-.LFB0
	.uleb128 0x1
	.byte	0x9c
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x1f
	.uleb128 0x1b
	.uleb128 0x1f
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7a
	.uleb128 0x19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.long	0x3c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.LFB0
	.quad	.LFE0-.LFB0
	.quad	.LFB1
	.quad	.LFE1-.LFB1
	.quad	0
	.quad	0
	.section	.debug_rnglists,"",@progbits
.Ldebug_ranges0:
	.long	.Ldebug_ranges3-.Ldebug_ranges2
.Ldebug_ranges2:
	.value	0x5
	.byte	0x8
	.byte	0
	.long	0
.LLRL0:
	.byte	0x7
	.quad	.LFB0
	.uleb128 .LFE0-.LFB0
	.byte	0x7
	.quad	.LFB1
	.uleb128 .LFE1-.LFB1
	.byte	0
.Ldebug_ranges3:
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF4:
	.string	"_Z3foov"
.LASF3:
	.string	"main"
.LASF2:
	.string	"GNU C++17 11.x"
	.section	.debug_line_str,"MS",@progbits,1
.LASF0:
	.string	"main.cpp"
.LASF1:
	.string	"/testRangeOffsetToRangeX"
	.ident	"GCC: (GNU) 11.x"
	.section	.note.GNU-stack,"",@progbits
