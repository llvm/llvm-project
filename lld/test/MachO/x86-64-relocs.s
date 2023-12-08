# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem -o %t %t.o
# RUN: llvm-objdump --no-print-imm-hex --section-headers --syms -d %t | FileCheck %s

# CHECK-LABEL: Sections:
# CHECK:       __data {{[0-9a-z]+}} [[#%x, DATA_ADDR:]]

# CHECK-LABEL: SYMBOL TABLE:
# CHECK:       [[#%x, F_ADDR:]] {{.*}} _f

# CHECK-LABEL: <_main>:
## Test X86_64_RELOC_BRANCH
# CHECK:       callq 0x[[#%x, F_ADDR]] <_f>
## Test extern (symbol) X86_64_RELOC_SIGNED
# CHECK:       leaq [[#%u, LOCAL_OFF:]](%rip), %rsi
# CHECK-NEXT:  [[#%x, DATA_ADDR - LOCAL_OFF]]
## Test non-extern (section) X86_64_RELOC_SIGNED
# CHECK:       leaq [[#%u, PRIVATE_OFF:]](%rip), %rsi
# CHECK-NEXT:  [[#%x, DATA_ADDR + 8 - PRIVATE_OFF]]

# RUN: llvm-objdump --section=__const --full-contents %t | FileCheck %s --check-prefix=NONPCREL
# NONPCREL:      Contents of section __DATA_CONST,__const:
# NONPCREL-NEXT: 100001000 08200000 01000000 08200000 01000000

.section __TEXT,__text
.globl _main, _f
_main:
  callq _f # X86_64_RELOC_BRANCH
  mov $0, %rax
  ret

_f:
  leaq _local(%rip), %rsi # Generates a X86_64_RELOC_SIGNED pcrel symbol relocation
  leaq L_.private(%rip), %rsi # Generates a X86_64_RELOC_SIGNED pcrel section relocation
  movq L_.ptr_1(%rip), %rsi
  ret

.data
## References to this generate a symbol relocation
_local:
  .quad 123
## References to this generate a section relocation
L_.private:
  .quad 123

.section __DATA,__const
## These generate X86_64_RELOC_UNSIGNED non-pcrel section relocations
L_.ptr_1:
  .quad L_.private
L_.ptr_2:
  .quad L_.private
