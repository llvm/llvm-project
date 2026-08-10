.section __TEXT,__cstring
.globl _hello_world, _hello_its_me, _print_hello

_hello_world:
.asciz "Hello world!\n"

_hello_its_me:
.asciz "Hello, it's me\n"

.text
_print_hello:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _hello_world(%rip), %rsi
  mov $13, %rdx # length of str
  syscall
  ret
