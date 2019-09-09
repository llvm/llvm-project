# REQUIRES: mips
# Check various cases of microMIPS - regular code cross-calls.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t-eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -position-independent -mattr=micromips \
# RUN:         %S/Inputs/mips-micro.s -o %t-eb-pic.o
# RUN: ld.lld -o %t-eb.exe %t-eb.o %t-eb-pic.o
# RUN: llvm-objdump -d -t -mattr=-micromips \
# RUN:              --no-show-raw-insn --print-imm-hex %t-eb.exe \
# RUN:   | FileCheck --check-prefixes=SYM,REG %s
# RUN: llvm-objdump -d -t -mattr=+micromips \
# RUN:              --no-show-raw-insn --print-imm-hex %t-eb.exe \
# RUN:   | FileCheck --check-prefixes=SYM,MICRO %s

# REG:        __start:
# REG-NEXT:      jalx 0x[[MIC:[0-9a-f]+]] <micro>
# REG-NEXT:      nop
# REG-NEXT:      jalx 0x[[FOOT:[0-9a-f]+]] <__microLA25Thunk_foo>

# REG:        __LA25Thunk_bar:
# REG-NEXT:      lui  $25, 0x2
# REG-NEXT:      j    0x[[BAR:[0-9a-f]+]] <bar>

# MICRO:      micro:
# MICRO-NEXT:    jalx 0x[[START:[0-9a-f]+]]
# MICRO-NEXT:    nop
# MICRO-NEXT:    jalx 0x[[BART:[0-9a-f]+]]

# MICRO:      __microLA25Thunk_foo:
# MICRO-NEXT:    lui  $25, 0x2
# MICRO-NEXT:    j    0x[[FOO:[0-9a-f]+]] <foo>

# REG:  [[FOOT]]   l     F .text  0000000e 0x80 __microLA25Thunk_foo
# REG:  [[BAR]]    g     F .text  00000000 bar
# REG:  [[MIC]]            .text  00000000 0x80 micro

# MICRO: [[BART]]  l     F .text  00000010 __LA25Thunk_bar
# MICRO: [[START]]         .text  00000000 __start
# MICRO: [[FOO]]   g     F .text  00000000 0x80 foo

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
