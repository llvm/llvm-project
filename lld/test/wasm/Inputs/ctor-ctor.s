	.section	.text.def,"",@
	.globl def
def:
	.functype	def () -> ()
	end_function

	.section	.text.test_ctor,"",@
	.globl test_ctor
test_ctor:
	.functype	test_ctor () -> ()
	end_function

	.section	.init_array,"",@
	.p2align	2
	.int32 test_ctor
