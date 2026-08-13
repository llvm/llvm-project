# REQUIRES: x86

## GCC uses PLTOFF64 in large-model TLSGD and TLSLD sequences. They do not
## match the fixed-size sequences that the linker can relax to IE or LE.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -e _start %t.o -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t.exe | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK:       leaq {{.*}}(%rip), %rdi
# CHECK-NEXT:  movabsq ${{.*}}, %rax
# CHECK-NEXT:  addq %rbx, %rax
# CHECK-NEXT:  callq *%rax
# CHECK-NEXT:  leaq {{.*}}(%rip), %rdi
# CHECK-NEXT:  movabsq ${{.*}}, %rax
# CHECK-NEXT:  addq %rbx, %rax
# CHECK-NEXT:  callq *%rax
# CHECK-NEXT:  leaq -4(%rax), %rax
# CHECK-NEXT:  retq

.section .ltext,"axl",@progbits
.globl _start
.type _start, @function
_start:
.Lpic:
  movabsq $_GLOBAL_OFFSET_TABLE_-.Lpic, %r11
  leaq .Lpic(%rip), %rbx
  addq %r11, %rbx

  leaq x@tlsgd(%rip), %rdi
  movabsq $__tls_get_addr@PLTOFF, %rax
  addq %rbx, %rax
  callq *%rax

  leaq x@tlsld(%rip), %rdi
  movabsq $__tls_get_addr@PLTOFF, %rax
  addq %rbx, %rax
  callq *%rax
  leaq x@dtpoff(%rax), %rax
  retq
.size _start, .-_start

.hidden __tls_get_addr
.globl __tls_get_addr
.type __tls_get_addr, @function
__tls_get_addr:
  retq
.size __tls_get_addr, .-__tls_get_addr

.section .tbss,"awT",@nobits
.hidden x
.globl x
x:
  .zero 4
