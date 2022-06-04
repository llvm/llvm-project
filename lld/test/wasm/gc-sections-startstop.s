# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o %s
# RUN: wasm-ld %t.o -o %t.wasm
# RUN: llvm-objdump -d --no-show-raw-insn %t.wasm | FileCheck %s

# FOO_MD symbol is not used directly, but is referenced through __start/__stop_foo_md
foo_md_size:
	.functype	foo_md_size () -> (i32)
	i32.const	__stop_foo_md
	i32.const	__start_foo_md
	i32.sub
	end_function

# CHECK: <foo_md_size>:
# CHECK-EMPTY:
# CHECK-NEXT: i32.const [[#STOP_ADDR:]]
# CHECK-NEXT: i32.const [[#STOP_ADDR - 4]]
# CHECK-NEXT: i32.sub

# All segments in concat_section section are marked as live.
concat_section_size:
	.functype	concat_section_size () -> (i32)
	i32.const	__stop_concat_section
	i32.const	__start_concat_section
	i32.sub
	end_function

# CHECK: <concat_section_size>:
# CHECK-EMPTY:
# CHECK-NEXT: i32.const [[#STOP_ADDR:]]
# CHECK-NEXT: i32.const [[#STOP_ADDR - 8]]
# CHECK-NEXT: i32.sub


# __start/__stop symbols don't retain invalid C name sections
invalid_name_section_size:
	.functype	invalid_name_section_size () -> (i32)
	i32.const	__stop_invalid.dot.name
	i32.const	__start_invalid.dot.name
	i32.sub
	end_function

# CHECK: <invalid_name_section_size>:
# CHECK-EMPTY:
# CHECK-NEXT: i32.const 0
# CHECK-NEXT: i32.const 0
# CHECK-NEXT: i32.sub


	.globl	_start
_start:
	.functype	_start () -> ()
	call	foo_md_size
	drop
	call	concat_section_size
	drop
	call	invalid_name_section_size
	drop
	end_function


	.section	foo_md,"",@
FOO_MD:
	.asciz	"bar"
	.size	FOO_MD, 4

	.size	__start_foo_md, 4
	.size	__stop_foo_md, 4


	.section	concat_section,"",@
concat_segment_1:
	.asciz	"xxx"
	.size	concat_segment_1, 4

concat_segment_2:
	.asciz	"yyy"
	.size	concat_segment_2, 4

	.size	__start_concat_section, 4
	.size	__stop_concat_section, 4


	.section	invalid.dot.name,"",@
invalid_name_section:
	.asciz	"fizz"
	.size	invalid_name_section, 5

	.weak	__start_invalid.dot.name
	.weak	__stop_invalid.dot.name
