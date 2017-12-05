.bss
.type sharedFoo,@object
.globl sharedFoo
sharedFoo:
.long 0
.size sharedFoo, 4

.type sharedBar,@object
.globl sharedBar
sharedBar:
.quad 0
.size sharedBar, 8
