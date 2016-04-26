# REQUIRES: x86, shell

# RUN: rm -rf %t.dir/repro
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2 -shared --as-needed --reproduce %t.dir/repro
# RUN: llvm-objdump -d %t.dir/repro/%t | FileCheck %s --check-prefix=DUMP
# RUN: cat %t.dir/repro/invocation.txt | FileCheck %s --check-prefix=INVOCATION

.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall

# DUMP: Disassembly of section .text:
# DUMP: _start:
# DUMP:        0:       48 c7 c0 3c 00 00 00    movq    $60, %rax
# DUMP:        7:       48 c7 c7 2a 00 00 00    movq    $42, %rdi
# DUMP:        e:       0f 05   syscall

# INVOCATION: lld {{.*}}reproduce.s{{.*}} -o {{.*}} -shared --as-needed --reproduce

# RUN: rm -rf %t.dir/repro2
# RUN: mkdir %t.dir/repro2
# RUN: not ld.lld %t -o %t2 --reproduce %t.dir/repro2 2>&1 | FileCheck --check-prefix=EDIR %s
# EDIR: --reproduce: can't create directory
