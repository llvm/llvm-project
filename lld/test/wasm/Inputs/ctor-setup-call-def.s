# Like Inputs/ctor-setup.s, except it calls `def` instead of `lib_func`,
# so it pulls in the .o file containing `ctor`.

	.section	.text._start,"",@
	.globl	_start
_start:
	.functype	_start () -> ()
	end_function

	.section	.text.setup,"",@
	.globl setup
setup:
	.functype	setup () -> ()
	call def
	end_function

	.section	.init_array,"",@
	.p2align	2
	.int32 setup

.functype       def () -> ()
