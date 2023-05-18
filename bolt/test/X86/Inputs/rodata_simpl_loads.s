.globl main
main:
	movzbl	s1(%rip),   %eax
	cmpb    s2(%rip),   %al
	movzbl  s1+1(%rip), %eax
	cmpb    s2+1(%rip), %al
	movzbl	s1+2(%rip), %eax
	cmpb	  s2+2(%rip), %al
	movzbl	s1+3(%rip), %eax
	cmpb	  s2+3(%rip), %al
	movl	  I1(%rip),   %eax
	addl	  I2(%rip),   %eax
	movl	  I2(%rip),   %eax

.rodata
"I1":
.long 6
"I2":
.long 67
"s1":
.string "ABC"
"s2":
.string "ABC"
