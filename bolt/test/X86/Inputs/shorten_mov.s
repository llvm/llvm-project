  .text
  .globl foo
  .type foo, @function
foo:
   movabs     $0x1111,%rax
   movabs     $0x1111,%rax
   movabs     $0x11112222,%rax
   movabs     $0x111112222,%rax
   movabs     $0xffffffffffffffff,%rax
   movabs     $0x7fffffffffffffff,%rax
   ret
   .size foo, .-foo
