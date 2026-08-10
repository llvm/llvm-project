.globl _bar,_dynamic,_foo_tag

.section .data,"",@
_bar:
.size _bar,4

_dynamic:
.size _dynamic,4

.tagtype _foo_tag i32
_foo_tag:
