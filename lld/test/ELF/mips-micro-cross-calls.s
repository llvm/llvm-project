# REQUIRES: mips
# Check various cases of microMIPS - regular code cross-calls.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t-eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -position-independent -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-eb-pic.o
# RUN: ld.lld -o %t-eb.exe %t-eb.o %t-eb-pic.o
# RUN: llvm-objdump -d -t --mattr=-micromips \
# RUN:              --no-show-raw-insn --print-imm-hex %t-eb.exe \
# RUN:   | FileCheck --check-prefix=REG %s
# RUN: llvm-objdump -d -t --mattr=+micromips \
# RUN:              --no-show-raw-insn --print-imm-hex %t-eb.exe \
# RUN:   | FileCheck --check-prefix=MICRO %s

# REG:  {{0*}}[[FOOT:[0-9a-f]+]]   l     F .text  0000000e 0x80 __microLA25Thunk_foo
# REG:  {{0*}}[[MIC:[0-9a-f]+]]    g       .text  00000000 0x80 micro
# REG:  {{0*}}[[BAR:[0-9a-f]+]]    g     F .text  00000000 bar

# REG:        <__start>:
# REG-NEXT:      jalx 0x[[MIC]] <micro>
# REG-NEXT:      nop
# REG-NEXT:      jalx 0x[[FOOT]] <__microLA25Thunk_foo>

# REG:        <__LA25Thunk_bar>:
# REG-NEXT:      lui  $25, 0x2
# REG-NEXT:      j    0x[[BAR]] <bar>

# MICRO: {{0*}}[[BART:[0-9a-f]+]]  l     F .text  00000010 __LA25Thunk_bar
# MICRO: {{0*}}[[START:[0-9a-f]+]] g       .text  00000000 __start
# MICRO: {{0*}}[[FOO:[0-9a-f]+]]   g     F .text  00000000 0x80 foo

# MICRO:      <micro>:
# MICRO-NEXT:    jalx 0x[[START]]
# MICRO-NEXT:    nop
# MICRO-NEXT:    jalx 0x[[BART]]

# MICRO:      <__microLA25Thunk_foo>:
# MICRO-NEXT:    lui  $25, 0x2
# MICRO-NEXT:    j    0x[[FOO]] <foo>

  .text
  .set nomicromips
  .global __start
__start:
  jal micro
  jal foo

  .set micromips
  .global micro
micro:
  jal __start
  jal bar
