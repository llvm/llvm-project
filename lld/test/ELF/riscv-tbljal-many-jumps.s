# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -mattr=zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -mattr=zcmt %s -o %t.rv64.o

# tbljal conversion
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 --defsym foo_1=0x150010 --defsym foo_3=0x150030 -o %t.rv64

# jump table
# RUN: llvm-objdump -h %t.rv32 | FileCheck --check-prefix=JUMPTABLE32 %s
# RUN: llvm-objdump -h %t.rv64 | FileCheck --check-prefix=JUMPTABLE64 %s

# JUMPTABLE32:  2 .riscv.jvt    00000080 {{.*}} TEXT
# JUMPTABLE64:  2 .riscv.jvt    00000100 {{.*}} TEXT


.global _start
.p2align 3
_start:
  tail foo
  tail foo
  tail foo
  tail foo
  tail foo_1
  tail foo_1
  tail foo_1
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_3
  tail foo_4
  tail foo_4
  tail foo_4
  tail foo_4
  tail foo_5
  tail foo_5
  tail foo_5
  tail foo_5
  tail foo_6
  tail foo_6
  tail foo_6
  tail foo_6
  tail foo_7
  tail foo_7
  tail foo_7
  tail foo_7
  tail foo_8
  tail foo_8
  tail foo_8
  tail foo_8
  tail foo_9
  tail foo_9
  tail foo_9
  tail foo_9

  tail foo_10
  tail foo_10
  tail foo_10
  tail foo_10
  tail foo_11
  tail foo_11
  tail foo_11
  tail foo_11
  tail foo_12
  tail foo_12
  tail foo_12
  tail foo_12
  tail foo_13
  tail foo_13
  tail foo_13
  tail foo_13
  tail foo_14
  tail foo_14
  tail foo_14
  tail foo_14
  tail foo_15
  tail foo_15
  tail foo_15
  tail foo_15
  tail foo_16
  tail foo_16
  tail foo_16
  tail foo_16
  tail foo_17
  tail foo_17
  tail foo_17
  tail foo_17
  tail foo_18
  tail foo_18
  tail foo_18
  tail foo_18
  tail foo_19
  tail foo_19
  tail foo_19
  tail foo_19

  tail foo_20
  tail foo_20
  tail foo_20
  tail foo_20
  tail foo_21
  tail foo_21
  tail foo_21
  tail foo_21
  tail foo_22
  tail foo_22
  tail foo_22
  tail foo_22
  tail foo_23
  tail foo_23
  tail foo_23
  tail foo_23
  tail foo_24
  tail foo_24
  tail foo_24
  tail foo_24
  tail foo_25
  tail foo_25
  tail foo_25
  tail foo_25
  tail foo_26
  tail foo_26
  tail foo_26
  tail foo_26
  tail foo_27
  tail foo_27
  tail foo_27
  tail foo_27
  tail foo_28
  tail foo_28
  tail foo_28
  tail foo_28
  tail foo_29
  tail foo_29
  tail foo_29
  tail foo_29

  tail foo_30
  tail foo_30
  tail foo_30
  tail foo_30
  tail foo_31
  tail foo_31
  tail foo_31
  tail foo_31
  tail foo_32
  tail foo_32
  tail foo_32
  tail foo_32
  tail foo_33
  tail foo_33
  tail foo_33
  tail foo_33
  tail foo_34
  tail foo_34
  tail foo_34
  tail foo_34
  tail foo_35
  tail foo_35
  tail foo_35
  tail foo_35
  tail foo_36
  tail foo_36
  tail foo_36
  tail foo_36
  tail foo_37
  tail foo_37
  tail foo_37
  tail foo_37
  tail foo_38
  tail foo_38
  tail foo_38
  tail foo_38
  tail foo_39
  tail foo_39
  tail foo_39
  tail foo_39


.space 16384

foo_3:
  nop

foo_4:
  nop

foo_5:
  nop

foo_6:
  nop

foo_7:
  nop

foo_8:
  nop

foo_9:
  nop

foo_10:
  nop
foo_11:
  nop
foo_12:
  nop
foo_13:
  nop
foo_14:
  nop
foo_15:
  nop
foo_16:
  nop
foo_17:
  nop
foo_18:
  nop
foo_19:
  nop

foo_20:
  nop
foo_21:
  nop
foo_22:
  nop
foo_23:
  nop
foo_24:
  nop
foo_25:
  nop
foo_26:
  nop
foo_27:
  nop
foo_28:
  nop
foo_29:
  nop

foo_30:
  nop
foo_31:
  nop
foo_32:
  nop
foo_33:
  nop
foo_34:
  nop
foo_35:
  nop
foo_36:
  nop
foo_37:
  nop
foo_38:
  nop
foo_39:
  nop
