# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem -o %t %t.o
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

# RUN: llvm-objdump --section=__const --full-contents -d %t | FileCheck %s --check-prefix=NONPCREL
# NONPCREL:      Contents of section __const:
# NONPCREL-NEXT: 100001000 d0030000 01000000 d0030000 01000000

.section __TEXT,__text
.globl _main, _f
_main:
  callq _f # X86_64_RELOC_BRANCH
  mov $0, %rax
  ret

_f:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _str(%rip), %rsi # Generates a X86_64_RELOC_SIGNED pcrel symbol relocation
  mov $21, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq L_.str(%rip), %rsi # Generates a X86_64_RELOC_SIGNED pcrel section relocation
  mov $15, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq L_.ptr_1_to_str(%rip), %rsi
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

.section __DATA,__const
## These generate X86_64_RELOC_UNSIGNED non-pcrel section relocations
L_.ptr_1_to_str:
  .quad L_.str
L_.ptr_2_to_str:
  .quad L_.str
