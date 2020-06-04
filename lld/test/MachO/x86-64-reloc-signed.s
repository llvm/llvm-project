# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump -D %t | FileCheck %s

# CHECK:      <_main>:
# CHECK-NEXT:   movl {{.*}}  # 100001000 <_s>
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movl {{.*}}  # 100001002 <_s+0x2>
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movb {{.*}}  # 100001000 <_s>
# CHECK-NEXT:   callq {{.*}}
# CHECK:      <__not_text>:
# CHECK-NEXT:   movl {{.*}}  # 100001005
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movl {{.*}}  # 100001007
# CHECK-NEXT:   callq {{.*}}
# CHECK-NEXT:   movb {{.*}}  # 100001005
# CHECK-NEXT:   callq {{.*}}

.section __TEXT,__text
.globl _main
_main:
  ## Symbol relocations
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

.section __TEXT,__not_text
  ## Section relocations. We intentionally put them in a separate section since
  ## the __text section typically starts at an address of zero in object files,
  ## and so does not fully exercise the relocation logic.
  movl $0x434241, L._s(%rip)  # X86_64_RELOC_SIGNED_4
  callq _f
  movl $0x44, L._s+2(%rip)    # X86_64_RELOC_SIGNED_2
  callq _f
  movb $0x45, L._s(%rip)      # X86_64_RELOC_SIGNED_1
  callq _f
  ret

.section __DATA,__data
.globl _s
_s:
  .space 5

## Create a new section to force the assembler to use a section relocation for
## the private symbol L._s. Otherwise, it will instead use a nearby non-private
## symbol to create a symbol relocation plus an addend.
.section __DATA,__foo
L._s:
  .space 5
