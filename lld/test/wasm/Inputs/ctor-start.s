	.globl _start
_start:
	.functype	_start () -> ()
	call lib_func
	end_function

	.functype	lib_func () -> ()
