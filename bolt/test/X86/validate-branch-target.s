## Test that BOLT errs when detecting the target 
## of a direct call/branch is a invalid instruction

# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t && cd %t
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o main.o
# RUN: %clang %cflags -pie -Wl,-q %t/main.o -o main.exe
# RUN: llvm-bolt %t/main.exe -o %t/main.exe.bolt -lite=0 2>&1 | FileCheck %s --check-prefix=CHECK-TARGETS

# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function external_corrcupt, an external branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}
# CHECK-TARGETS: BOLT-WARNING: corrupted control flow detected in function internal_corrcupt, an internal branch/call targets an invalid instruction at address 0x{{[0-9a-f]+}}


.globl	internal_corrcupt
.type	internal_corrcupt,@function
.align	16
internal_corrcupt:
	leaq	.Lopts_1(%rip),%rax
	addq	$25,%rax
	.byte	0xf3,0xc3
.L8xchar_1:
	addq	$12,%rax
.Ldone_1:
	.byte	0xf3,0xc3
.align	64
.Lopts_1:
.byte	114,1,52,40,56,120,44,105,110,116,41,0  # data '114' will be disassembled as 'jb', check for internal branch: jb + 0x1
.align	64
.size	internal_corrcupt,.-internal_corrcupt


.globl	external_corrcupt
.type	external_corrcupt,@function
.align	16
external_corrcupt:
	leaq	.Lopts_2(%rip),%rax
	addq	$25,%rax
	.byte	0xf3,0xc3
.L8xchar_2:
	addq	$12,%rax
.Ldone_2:
	.byte	0xf3,0xc3
.align	64
.Lopts_2:
.byte	114,99,52,40,56,120,44,99,104,97,114,41,0  # data '114' will be disassembled as 'jb', check for external branch: jb + 0x63
.align	64
.size	external_corrcupt,.-external_corrcupt
