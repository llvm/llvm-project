	.text

.macro define_func name, symbol
	.globl	\name
	.type	\name,@function
\name:
	.functype	\name () -> (i32)
	i32.const 0
	i32.load \symbol
	end_function
.endm

define_func f1, foo
define_func f2, foo
define_func f3, foo
define_func f4, foo
define_func f5, foo
define_func f6, foo
define_func f7, foo
define_func f8, foo
define_func f9, foo
define_func f10, foo
define_func f11, foo
define_func f12, foo
define_func f13, foo
define_func f14, foo
define_func f15, foo
define_func f16, foo
define_func f17, foo
define_func f18, foo
define_func f19, foo
define_func f20, foo
define_func f21, foo
define_func f22, foo
define_func f23, foo
define_func f24, foo
define_func f25, foo
define_func f26, foo
define_func f27, foo
define_func f28, foo
define_func f29, foo
define_func f30, foo
define_func f31, foo
define_func f32, foo
define_func f33, foo
define_func f34, foo
define_func f35, foo
define_func f36, foo
define_func f37, foo
define_func f38, foo
define_func f39, foo
define_func f40, foo
define_func f41, foo
define_func f42, foo
define_func f43, foo
define_func f44, foo
define_func f45, foo
define_func f46, foo
define_func f47, foo
define_func f48, foo
define_func f49, foo
define_func f50, foo
define_func f51, foo
define_func f52, foo
define_func f53, foo
define_func f54, foo
define_func f55, foo
define_func f56, foo
define_func f57, foo
define_func f58, foo
define_func f59, foo
define_func f60, foo
define_func f61, foo
define_func f62, foo
define_func f63, foo
define_func f64, foo
define_func f65, foo
define_func f66, foo
define_func f67, foo
define_func f68, foo
define_func f69, foo
define_func f70, foo
define_func f71, foo
define_func f72, foo
define_func f73, foo
define_func f74, foo
define_func f75, foo
define_func f76, foo
define_func f77, foo
define_func f78, foo
define_func f79, foo
define_func f80, foo
define_func f81, foo
define_func f82, foo
define_func f83, foo
define_func f84, foo
define_func f85, foo
define_func f86, foo
define_func f87, foo
define_func f88, foo
define_func f89, foo
define_func f90, foo
define_func f91, foo
define_func f92, foo
define_func f93, foo
define_func f94, foo
define_func f95, foo
define_func f96, foo
define_func f97, foo
define_func f98, foo
define_func f99, foo
define_func f100, foo
define_func f101, foo
define_func f102, foo
define_func f103, foo
define_func f104, foo
define_func f105, foo
define_func f106, foo
define_func f107, foo
define_func f108, foo
define_func f109, foo
define_func f110, foo
define_func f111, foo
define_func f112, foo
define_func f113, foo
define_func f114, foo
define_func f115, foo
define_func f116, foo
define_func f117, foo
define_func f118, foo
define_func f119, foo
define_func f120, foo
define_func f121, foo
define_func f122, foo
define_func f123, foo
define_func f124, foo
define_func f125, foo
define_func f126, foo
define_func f127, foo
define_func f128, g0
define_func f129, g0

	.section	.data.g0,"",@
	.globl	g0
	.p2align	2
g0:
	.int32	1
	.size	g0, 4

	.section	.data.foo,"",@
	.globl	foo
	.p2align	2
foo:
	.int32	1
	.size	foo, 4
