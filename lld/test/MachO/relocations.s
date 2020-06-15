# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -arch x86_64 -o %t %t.o
# RUN: llvm-objdump --section-headers --syms -d %t | FileCheck %s

# CHECK-LABEL: Sections:
# CHECK:       __cstring {{[0-9a-z]+}} [[#%x, CSTRING_ADDR:]]

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x, F_ADDR:]] {{.*}} _f

# CHECK-LABEL: <_main>:
## Test X86_64_RELOC_BRANCH
# CHECK:       callq 0x[[#%x, F_ADDR]] <_f>
## Test extern (symbol) X86_64_RELOC_SIGNED
# CHECK:       leaq [[#%u, STR_OFF:]](%rip), %rsi
# CHECK-NEXT:  [[#%x, CSTRING_ADDR - STR_OFF]]
## Test non-extern (section) X86_64_RELOC_SIGNED
# CHECK:       leaq [[#%u, LSTR_OFF:]](%rip), %rsi
# CHECK-NEXT:  [[#%x, CSTRING_ADDR + 22 - LSTR_OFF]]

.section __TEXT,__text
.globl _main, _f
_main:
  callq _f
  mov $0, %rax
  ret

_f:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _str(%rip), %rsi
  mov $21, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq L_.str(%rip), %rsi
  mov $15, %rdx # length of str
  syscall
  ret

.section __TEXT,__cstring
## References to this generate a symbol relocation
_str:
  .asciz "Local defined symbol\n"
## References to this generate a section relocation
L_.str:
  .asciz "Private symbol\n"
