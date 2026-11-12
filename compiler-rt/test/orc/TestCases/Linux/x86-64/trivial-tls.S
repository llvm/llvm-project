// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic ELF TLS work by adding together TLSs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (.tbss) and non-zero-initialized
// (.tdata) sections.

	.text
	.file	"tlstest.cpp"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
# %bb.0:                                # %entry
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	data16
	leaq	x@TLSGD(%rip), %rdi
	data16
	data16
	rex64
	callq	__tls_get_addr@PLT
	movl	(%rax), %eax
	movl	%eax, -24(%rbp)                 # 4-byte Spill
	data16
	leaq	y@TLSGD(%rip), %rdi
	data16
	data16
	rex64
	callq	__tls_get_addr@PLT
	movq	%rax, %rcx
	movl	-24(%rbp), %eax                 # 4-byte Reload
	movl	(%rcx), %ecx
	addl	%ecx, %eax
	movl	%eax, -20(%rbp)                 # 4-byte Spill
	data16
	leaq	z@TLSGD(%rip), %rdi
	data16
	data16
	rex64
	callq	__tls_get_addr@PLT
	movq	%rax, %rcx
	movl	-20(%rbp), %eax                 # 4-byte Reload
	movl	(%rcx), %ecx
	addl	%ecx, %eax
	addq	$32, %rsp
	popq	%rbp
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        # -- End function
	.type	x,@object                       # @x
	.section	.tbss,"awT",@nobits
	.globl	x
	.p2align	2
x:
	.long	0                               # 0x0
	.size	x, 4

	.type	y,@object                       # @y
	.section	.tdata,"awT",@progbits
	.globl	y
	.p2align	2
y:
	.long	1                               # 0x1
	.size	y, 4

	.type	z,@object                       # @z
	.globl	z
	.p2align	2
z:
	.long	4294967295                      # 0xffffffff
	.size	z, 4

	.section	".note.GNU-stack","",@progbits
	.addrsig
