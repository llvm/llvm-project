.globl foo, bar
.type foo, @object
.size foo, 4
foo:
.long 1

.weak bar
.type bar, @object
.size bar, 4
bar:
.long 2
