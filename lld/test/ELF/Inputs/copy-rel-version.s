.data
.global foo_v1
.symver foo_v1, foo@v1, remove
.type foo_v1, @object
.size foo_v1, 4

.global foo_v2
.symver foo_v2, foo@v2, remove
.type foo_v2, @object
.size foo_v2, 8

.global foo
.symver foo, foo@@@v3
.type foo, @object
.size foo, 12

foo_v1:
foo_v2:
foo:
.int 0
.int 0
.int 0
