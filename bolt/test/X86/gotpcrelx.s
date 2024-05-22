# REQUIRES: system-linux

## Check that BOLT correctly handles different types of instructions with
## R_X86_64_GOTPCRELX or R_X86_64_REX_GOTPCRELX relocations and different
## kinds of handling of the relocation by the linker (no relaxation, pic, and
## non-pic).

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux \
# RUN:   -relax-relocations %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: ld.lld %t.o -o %t.pie.exe -q -pie
# RUN: ld.lld %t.o -o %t.no-relax.exe -q --no-relax
# RUN: llvm-bolt %t.exe --relocs -o %t.out --print-cfg --print-only=_start \
# RUN:   |& FileCheck --check-prefix=BOLT %s
# RUN: llvm-bolt %t.pie.exe -o %t.null --print-cfg --print-only=_start \
# RUN:   |& FileCheck --check-prefix=PIE-BOLT %s
# RUN: llvm-bolt %t.no-relax.exe -o %t.null --print-cfg --print-only=_start \
# RUN:   |& FileCheck --check-prefix=NO-RELAX-BOLT %s
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex \
# RUN:   %t.out | FileCheck --check-prefix=DISASM %s

## Relocate foo only and check that code references from _start (that is
## otherwise preserved) are updated.

# RUN: llvm-bolt %t.exe --relocs -o %t.lite.out --funcs=foo
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex \
# RUN:   %t.lite.out | FileCheck --check-prefix=DISASM %s

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# DISASM: <_start>:

                      call *foo@GOTPCREL(%rip)
# NO-RELAX-BOLT:      callq *{{.*}}(%rip)
# BOLT:               callq foo
# PIE-BOLT:           callq foo
# DISASM-NEXT:        callq 0x[[#%x,ADDR:]]

                      movq foo@GOTPCREL(%rip), %rdi
# NO-RELAX-BOLT-NEXT: movq {{.*}}(%rip), %rdi
# BOLT-NEXT:          leaq foo(%rip), %rdi
# PIE-BOLT-NEXT:      leaq foo(%rip), %rdi
# DISASM-NEXT:        leaq {{.*}}(%rip), %rdi # 0x[[#ADDR]]

                      movl foo@GOTPCREL+4(%rip), %edi
# NO-RELAX-BOLT-NEXT: movl {{.*}}(%rip), %edi
# BOLT-NEXT:          movl {{.*}}(%rip), %edi
# PIE-BOLT-NEXT:      movl {{.*}}(%rip), %edi
# DISASM-NEXT:        movl {{.*}}(%rip), %edi

                      test %rdi, foo@GOTPCREL(%rip)
# NO-RELAX-BOLT-NEXT: testq %rdi, DATA{{.*}}(%rip)
# BOLT-NEXT:          testq $foo, %rdi
# PIE-BOLT-NEXT:      testq %rdi, DATA{{.*}}(%rip)
# DISASM-NEXT:        testq $0x[[#ADDR]], %rdi

                      cmpq foo@GOTPCREL(%rip), %rax
# NO-RELAX-BOLT-NEXT: cmpq DATA{{.*}}(%rip), %rax
# BOLT-NEXT:          cmpq $foo, %rax
# PIE-BOLT-NEXT:      cmpq DATA{{.*}}(%rip), %rax
# DISASM-NEXT:        cmpq $0x[[#ADDR]], %rax

                      jmp *foo@GOTPCREL(%rip)
# NO-RELAX-BOLT-NEXT: jmpq *DATA{{.*}}(%rip)
# BOLT-NEXT:          jmp foo
# PIE-BOLT-NEXT:      jmp foo
# DISASM-NEXT:        jmp 0x[[#ADDR]]

# DISASM: [[#ADDR]] <foo>:

  ret
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo
