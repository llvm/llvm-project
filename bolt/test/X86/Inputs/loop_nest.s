  .globl main
main:
	jmp	a
b:
	callq	foo
a:
	jle	b

  .globl foo
foo:
	jmp	d
e:
	je	f
	cmpl	$1, g
f:
	jmp	j
h:
	cmpl	$1, 0
j:
	jle	h
g:
	cmpl	$1, 0
d:
	jle	e
i:
	retq
# FDATA: 1 foo #i# 1 main #a# 0 1
