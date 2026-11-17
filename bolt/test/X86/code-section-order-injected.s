# Check the physical order of BOLT-created main, cold, and injected sections in
# both layout directions. The reverse layout exercises the previous
# non-transitive ordering between multiple cold sections and the injected
# section.

# REQUIRES: x86_64-linux, bolt-runtime

# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start -o %t %t.o
# RUN: llvm-bolt %t -o %t.forward --lite=0 --hugify --split-functions \
# RUN:   --split-strategy=all --split-all-cold
# RUN: llvm-nm --defined-only --numeric-sort %t.forward \
# RUN:   | FileCheck --check-prefix=FORWARD %s
# RUN: llvm-bolt %t -o %t.reverse --lite=0 --hugify --split-functions \
# RUN:   --split-strategy=all --split-all-cold --hot-functions-at-end
# RUN: llvm-nm --defined-only --numeric-sort %t.reverse \
# RUN:   | FileCheck --check-prefix=REVERSE %s

# FORWARD: {{[0-9a-f]+}} T _start{{$}}
# FORWARD: {{[0-9a-f]+}} t _start.cold.0{{$}}
# FORWARD: {{[0-9a-f]+}} t _start.cold.1{{$}}
# FORWARD: {{[0-9a-f]+}} t _start.cold.2{{$}}
# FORWARD: {{[0-9a-f]+}} t __bolt_hugify_start_program{{$}}

# REVERSE: {{[0-9a-f]+}} t __bolt_hugify_start_program{{$}}
# REVERSE: {{[0-9a-f]+}} t _start.cold.2{{$}}
# REVERSE: {{[0-9a-f]+}} t _start.cold.1{{$}}
# REVERSE: {{[0-9a-f]+}} t _start.cold.0{{$}}
# REVERSE: {{[0-9a-f]+}} T _start{{$}}

  .text
  .globl _start
  .type _start,@function
_start:
  callq target
  testl %edi, %edi
  je .Lone
  # Keep each outlined fragment large enough to pass x86 split profitability.
  .rept 32
  addl $1, %edi
  .endr
  jmp .Ldone
.Lone:
  .rept 32
  subl $1, %edi
  .endr
.Ldone:
  retq
  .size _start, .-_start

  .globl target
  .type target,@function
target:
  retq
  .size target, .-target
