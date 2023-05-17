# REQUIRES: system-linux

# RUN: llvm-mc -dwarf-version=5 -filetype=obj -triple x86_64-unknown-linux %s -o %t1.o
# RUN: %clang %cflags -dwarf-5 %t1.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.exe | FileCheck --check-prefix=PRECHECK %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %t.bolt | FileCheck --check-prefix=POSTCHECK %s

# This test checks we correctly insert DW_AT_addr_base, when converting DW_AT_low_pc into DW_AT_ranges.
# PRECHECK-NOT: DW_AT_addr_base

# POSTCHECK: DW_AT_ranges [DW_FORM_rnglistx]
# POSTCHECK: DW_AT_rnglists_base [DW_FORM_sec_offset]	(0x0000000c)
# POSTCHECK-NEXT: DW_AT_addr_base [DW_FORM_sec_offset]	(0x00000008)

# int main() {
#    return 0;
# }


	.file	"main.cpp"
	.text
.Ltext0:
	.globl	main
	.type	main, @function
main:
.LFB0:
	.file 1 "main.cpp"
	.loc 1 1 12
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	.loc 1 2 10
	movl	$0, %eax
	.loc 1 3 1
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
.Letext0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x50
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x1
	.long	.LASF0
	.byte	0x21
	.long	.LASF1
	.long	.LASF2
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.long	.Ldebug_line0
	.uleb128 0x2
	.long	.LASF3
	.byte	0x1
	.byte	0x1
	.byte	0x5
	.long	0x4c
	.quad	.LFB0
	.quad	.LFE0-.LFB0
	.uleb128 0x1
	.byte	0x9c
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.string	"int"
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
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
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
	.uleb128 0x7a
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
	.byte	0
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF0:
	.string	"GNU C++14 8.5.0 20210514 (Red Hat 8.5.0-3) -mtune=generic -march=x86-64 -g2 -gdwarf-5"
.LASF1:
	.string	"main.cpp"
.LASF3:
	.string	"main"
.LASF2:
	.string	"."
	.ident	"GCC: (GNU) 8.5.0 20210514 (Red Hat 8.5.0-3)"
	.section	.note.GNU-stack,"",@progbits
