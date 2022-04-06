.globl bar
bar:
	movq	$0, (%rbp)
	movl	$4, %edi
	call	__cxa_allocate_exception
	movl	$0, (%rax)
	movl	$0, %edx
	movl	$_ZTIi, %esi
	movq	%rax, %rdi
	call	__cxa_throw
  movq	$17, 8

.globl	foo
foo:
.LFB1:
	.cfi_startproc
	.cfi_personality 0x3,__gxx_personality_v0
	.cfi_lsda 0x3,.LLSDA1
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%rbx
	subq	$24, %rsp
	movq	%rdi, -24(%rbp)
  incq  -24(%rbp)
  jmp   .L1
  decq  (%rbp)
.L1: incq  -24(%rbp)
  cmpq  $2,-24(%rbp)
  jne   .L3
  jmp   .L4
  decq  (%rbp)
.L3: incq  -24(%rbp)
.L4: incq  -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
.LEHB0:
	call	bar
.LEHE0:
	movq	%rax, %rbx
	.L5:
	movq	%rbx, %rax
	jmp	.L8
.L7:
	movq	%rax, %rdi
	call	__cxa_begin_catch
  incq  -24(%rbp)
  jmp   .LP1
  decq  (%rbp)
.LP1: incq  -24(%rbp)
  cmpq  $2,-24(%rbp)
  jne   .LP2
  jmp   .LP3
  decq  (%rbp)
.LP2: incq  -24(%rbp)
.LP3: incq  -24(%rbp)
	movq	-24(%rbp), %rbx
.LEHB1:
	call	__cxa_end_catch
.LEHE1:
	jmp	.L5
.L8:
	movq	-8(%rbp), %rbx
	leave
	.cfi_endproc

.section	.gcc_except_table,"a",@progbits
	.LLSDA1:
	.byte	0xff
	.byte	0x3
	.uleb128 .LLSDATT1-.LLSDATTD1
.LLSDATTD1:
	.byte	0x1
	.uleb128 .LLSDACSE1-.LLSDACSB1
.LLSDACSB1:
	.uleb128 .LEHB0-.LFB1
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L7-.LFB1
	.uleb128 0x1
	.LLSDACSE1:
	.LLSDATT1:

	.text
	.globl	_start, function
_start:
	.cfi_startproc
	ud2
	.cfi_endproc
