.text

.global dummy
.type   dummy, @function
dummy:  xor %rax, %rax
        retq
        .size dummy, .-dummy

.globl test_func
.type  test_func, @function

test_func:
     	leaq	-0x4(%rsi), %rax
     	cmpq	$0x4, %rax
     	jbe	.LFT22210

     	leaq	-0x9(%rsi), %rcx
     	cmpq	$0x7, %rcx
     	jbe	.LFT22211

     	leaq	-0x11(%rsi), %r8
     	cmpq	$0xf, %r8
     	ja	.Ltmp88386
     	jmp	.LFT22212

.LFT22210:
     	movq	(%rdi), %rax
     	retq

.Ltmp88386:
     	cmpq	$0x20, %rsi
     	jbe	.Ltmp88387
     	jmp	.LFT22213

.LFT22211:
     	xorq	%r11, %rax
     	retq

.LFT22212:
     	imulq	%rsi, %rax
     	retq

.LFT22213:
     	jmp	dummy

.Ltmp88387:
     	movabsq	$-0x651e95c4d06fbfb1, %rax
     	xorq	%rdx, %rax
     	testq	%rsi, %rsi
     	jne	.LFT22214

.Ltmp88388:
     	retq

.LFT22214:
     	imulq	%rcx, %rax
     	jmp	.Ltmp88388

    .size   test_func, .-test_func

.globl  main
.type   main, @function
main:
        xorl    %eax, %eax
	    retq
        .size   main, .-main
