# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK:      <_main>:
# CHECK-NEXT:   movl {{.*}}  # 2000 <_s>
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movl {{.*}}  # 2002 <_s+0x2>
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movb {{.*}}  # 2000 <_s>
# CHECK-NEXT:   callq {{.*}}

.section __TEXT,__text
.globl _main
_main:
  movl $0x434241, _s(%rip)  # X86_64_RELOC_SIGNED_4
  callq _f
  movl $0x44, _s+2(%rip)    # X86_64_RELOC_SIGNED_2
  callq _f
  movb $0x45, _s(%rip)      # X86_64_RELOC_SIGNED_1
  callq _f
  xorq %rax, %rax
  ret

_f:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq _s(%rip), %rsi
  mov $3, %rdx # length
  syscall
  ret

.section __DATA,__data
.globl _s
_s:
  .space 5
