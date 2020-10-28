# Like Inputs/ctor-start.s, except it calls `lib_func` from a ctor
# instead of from `_start`.

	.globl	_start
_start:
	.functype	_start () -> ()
	end_function

	.globl	setup
setup:
	.functype	setup () -> ()
	call	lib_func
	end_function

	.section	.init_array,"",@
	.p2align	2
	.int32	setup

        .functype       lib_func () -> ()
