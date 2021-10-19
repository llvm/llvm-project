# REQUIRES: x86
# RUN: llvm-mc %s -filetype=obj -triple=x86_64-windows-msvc -o %t.obj
# RUN: lld-link %t.obj -export:foo -export:bar -dll -noentry -out:%t.dll -verbose 2>&1 | FileCheck %s
# RUN: llvm-readobj --sections %t.dll | FileCheck %s --check-prefix=TEXT

# The order of the pdata and xdata sections here shouldn't matter. We should
# still replace bar with foo.

# CHECK: ICF needed {{.*}} iterations
# CHECK: Selected foo
# CHECK: Removed bar

# We should only have five bytes of text.
# TEXT: Name: .text
# TEXT-NEXT: Size: 0x5

	.section	.text,"xr",discard,foo
	.globl	foo
foo:
	pushq %rbx
	pushq %rdi
	popq %rdi
	popq %rbx
        retq


.section .pdata,"r",associative,foo
.long foo
.long 5
.long foo_xdata@IMGREL

.section .xdata,"r",associative,foo
foo_xdata:
.long 42

	.section	.text,"xr",discard,bar
	.globl	bar
bar:
	pushq %rbx
	pushq %rdi
	popq %rdi
	popq %rbx
        retq

.section .xdata,"r",associative,bar
bar_xdata:
.long 42

.section .pdata,"r",associative,bar
.long bar
.long 5
.long bar_xdata@IMGREL
