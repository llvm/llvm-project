# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump --section-headers --syms -d %t | FileCheck %s

# CHECK-LABEL: Sections:
# CHECK:       __cstring {{[0-9a-z]+}} [[#%x, CSTRING_ADDR:]]

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x, F_ADDR:]] {{.*}} _f

# CHECK-LABEL: <_main>:
## Test X86_64_RELOC_BRANCH
# CHECK:       callq 0x[[#%x, F_ADDR]] <_f>
## Test X86_64_RELOC_SIGNED
# CHECK:       leaq [[#%u, STR_OFF:]](%rip), %rsi
# CHECK-NEXT:  [[#%x, CSTRING_ADDR - STR_OFF]]

.section __TEXT,__text
.globl _main, _f
_main:
  callq _f
  mov $0, %rax
  ret

_f:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq str(%rip), %rsi
  mov $13, %rdx # length of str
  syscall
  ret

.section __TEXT,__cstring
str:
  .asciz "Hello world!\n"
