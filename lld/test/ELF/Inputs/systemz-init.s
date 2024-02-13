// glibc < 2.39 used to align .init and .fini code at a 4-byte boundary.
// This file aims to recreate that behavior.
	.section        .init,"ax",@progbits
	.align	4
	lg %r4, 272(%r15)
