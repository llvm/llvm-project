# REQUIRES: x86
# RUN: llvm-mc %s -filetype=obj -triple=x86_64-windows-msvc -o %t.obj
# RUN: lld-link %t.obj -export:foo -export:bar -dll -noentry -out:%t.dll -merge:.xdata=.xdata -verbose 2>&1 | FileCheck %s
# RUN: llvm-readobj --sections %t.dll | FileCheck %s --check-prefix=XDATA

# Test xdata can be merged when text and pdata differ. This test is structured
# so that xdata comes after pdata, which makes xdata come before pdata in the
# assocChildren linked list.

# CHECK: ICF needed {{.*}} iterations
# CHECK: Selected
# CHECK: Removed

# XDATA:         Name: .xdata
# XDATA-NEXT:    VirtualSize: 0x4

	.section	.text,"xr",discard,foo
	.globl	foo
foo:
	pushq %rax
	popq %rax
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
	pushq %rcx
	popq %rcx
        retq

.section .pdata,"r",associative,bar
.long bar
.long 5
.long bar_xdata@IMGREL

.section .xdata,"r",associative,bar
bar_xdata:
.long 42
