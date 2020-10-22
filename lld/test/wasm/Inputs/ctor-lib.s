	.section	.text.lib_func,"",@
	.globl	lib_func
lib_func:
	.functype	lib_func () -> ()
	end_function

	.section	.text.unused_lib_func,"",@
	.globl unused_lib_func
unused_lib_func:
	.functype	unused_lib_func () -> ()
	call def
	end_function

	.functype	def () -> ()
