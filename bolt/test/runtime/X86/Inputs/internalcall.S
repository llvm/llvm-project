// This is reduced test case for BOLT containing an internal call based on
// GetCoreDump (from google core dumper).

  .text
  .globl getCallback
  .type getCallback, %function
getCallback:
  .cfi_startproc
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%rbx
	subq	$288, %rsp
	callq	.Lnext_instr
.Lnext_instr:
	popq	%rax
  addq  $17, %rax
	addq	$288, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	retq
.Lweird_callback:
  mov $0xDEADBEEF, %rax
  retq
  .cfi_endproc


// This one is inspired by:
//
//  e8 11 00 00 00  callq   17 <ListerThread+0x569>
//  0f 1f 84 00 00 00 00 00         nopl    (%rax,%rax)
//  48 c7 c0 0f 00 00 00    movq    $15, %rax
//  0f 05   syscall
//  58      popq    %rax
//
  .globl getCallback2
  .type getCallback2, %function
getCallback2:
  .cfi_startproc
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%rbx
	subq	$288, %rsp
  movq $3, %rbx
.Lheader:
  cmpq $0, %rbx
  jz .Lunwind
	callq	.Lbb
  nopl (%rax, %rax)
  mov $0xDEADBEEF, %rax
  retq
.Lunwind:
	addq	$288, %rsp
	popq	%rbx
	popq	%r12
  jmp .Lend
.Lbb:
	popq	%rax
  add $4, %rax
  decq %rbx
  jmp .Lheader
.Lend:
	popq	%rbp
	retq
  .cfi_endproc

// This case emulates pseudo-inlined functions found in Intel MKL library.
  .globl getCallback3
  .type getCallback3, %function
getCallback3:
  .cfi_startproc
  pushq %rbp
  movq %rsp, %rbp
  callq getCallback_inlined
  jmp .L2end
  nop
getCallback_inlined:
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%rbx
	subq	$288, %rsp
	callq	.L2next_instr
.L2next_instr:
	popq	%rax
  addq  $17, %rax
	addq	$288, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	retq
.L2weird_callback:
  mov $0xDEADBEEF, %rax
  retq

.L2end:
  popq %rbp
  retq
  .cfi_endproc
