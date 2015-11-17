	.file	"x.c"
	.hidden	_nl_default_default_domain
	.globl	_nl_default_default_domain
	.section	.rodata._nl_default_default_domain,"ams",@progbits,1
	.type	_nl_default_default_domain, @object
	.size	_nl_default_default_domain, 9
_nl_default_default_domain:
	.string	"messages"
	.hidden	_nl_current_default_domain
	.globl	_nl_current_default_domain
	.section	.data._nl_current_default_domain,"aw",@progbits
	.align 8
	.type	_nl_current_default_domain, @object
	.size	_nl_current_default_domain, 8
_nl_current_default_domain:
	.quad	_nl_default_default_domain
	.globl	_nl_default_default_dirname
	.section	.rodata._nl_default_default_dirname,"ams",@progbits,1
	.type	_nl_default_default_dirname, @object
	.size	_nl_default_default_dirname, 11
_nl_default_default_dirname:
	.string	"/usr/local"
	.ident	"GCC: (Ubuntu 4.8.1-2ubuntu1~10.04.1) 4.8.1"
	.section	.note.GNU-stack,"",@progbits
