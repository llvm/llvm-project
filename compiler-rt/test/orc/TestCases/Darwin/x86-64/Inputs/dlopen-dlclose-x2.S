// Runs a sequence of dlopen, dlclose, dlopen, dlclose on a library "inits".
// This is intended as a standard harness for testing constructor / destructor
// behavior in the context of a full dlclose and then re-dlopen'ing of the
// inits library.
//
// Compiled from:
//
// int main(int argc, char *argv[]) {
//  printf("entering main\n");
//  void *H = dlopen("inits", 0);
//  if (!H) {
//    printf("failed\n");
//    return -1;
//  }
//  if (dlclose(H) == -1) {
//    printf("failed\n");
//    return -1;
//  }
//  H = dlopen("inits", 0);
//  if (!H) {
//    printf("failed\n");
//    return -1;
//  }
//  if (dlclose(H) == -1) {
//    printf("failed\n");
//    return -1;
//  }
//  printf("leaving main\n");
//  return 0;
//}

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 13, 0	sdk_version 13, 0
	.globl	_main
	.p2align	4, 0x90
_main:

	pushq	%r14
	pushq	%rbx
	pushq	%rax
	leaq	L_str(%rip), %rdi
	callq	_puts
	leaq	L_.str.1(%rip), %rdi
	xorl	%esi, %esi
	callq	_dlopen
	movl	$-1, %ebx
	leaq	L_str.8(%rip), %r14
	testq	%rax, %rax
	je	LBB0_4

	movq	%rax, %rdi
	callq	_dlclose
	cmpl	$-1, %eax
	je	LBB0_4

	leaq	L_.str.1(%rip), %rdi
	xorl	%esi, %esi
	callq	_dlopen
	testq	%rax, %rax
	je	LBB0_4

	movq	%rax, %rdi
	callq	_dlclose
	xorl	%ebx, %ebx
	cmpl	$-1, %eax
	sete	%bl
	leaq	L_str.8(%rip), %rax
	leaq	L_str.6(%rip), %r14
	cmoveq	%rax, %r14
	negl	%ebx
LBB0_4:
	movq	%r14, %rdi
	callq	_puts
	movl	%ebx, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%r14
	retq

	.section	__TEXT,__cstring,cstring_literals
L_.str.1:
	.asciz	"inits"
L_str:
	.asciz	"entering main"
L_str.6:
	.asciz	"leaving main"
L_str.8:
	.asciz	"failed"

.subsections_via_symbols
