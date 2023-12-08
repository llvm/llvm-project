.section __TEXT,__cstring
.globl _goodbye_world, _print_goodbye

_goodbye_world:
.asciz "Goodbye world!\n"

.text
_print_goodbye:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _goodbye_world(%rip), %rsi
  mov $15, %rdx # length of str
  syscall
  ret
