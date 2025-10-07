# REQUIRES: riscv

## Test table jump with many targets filling the cm.jt table (32 entries).
## Verify the .riscv.jvt section size accounts for all entries.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt -asm-macro-max-nesting-depth=33 %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+zcmt -asm-macro-max-nesting-depth=33 %s -o %t.rv64.o

## Use --defsym for targets beyond c.j range so they can only be relaxed via table jump.
# RUN: ld.lld %t.rv32.o --relax-tbljal \
# RUN:   --defsym t0=0x100000 --defsym t1=0x100010 --defsym t2=0x100020 --defsym t3=0x100030 \
# RUN:   --defsym t4=0x100040 --defsym t5=0x100050 --defsym t6=0x100060 --defsym t7=0x100070 \
# RUN:   --defsym t8=0x100080 --defsym t9=0x100090 --defsym t10=0x1000a0 --defsym t11=0x1000b0 \
# RUN:   --defsym t12=0x1000c0 --defsym t13=0x1000d0 --defsym t14=0x1000e0 --defsym t15=0x1000f0 \
# RUN:   --defsym t16=0x100100 --defsym t17=0x100110 --defsym t18=0x100120 --defsym t19=0x100130 \
# RUN:   --defsym t20=0x100140 --defsym t21=0x100150 --defsym t22=0x100160 --defsym t23=0x100170 \
# RUN:   --defsym t24=0x100180 --defsym t25=0x100190 --defsym t26=0x1001a0 --defsym t27=0x1001b0 \
# RUN:   --defsym t28=0x1001c0 --defsym t29=0x1001d0 --defsym t30=0x1001e0 --defsym t31=0x1001f0 \
# RUN:   -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal \
# RUN:   --defsym t0=0x100000 --defsym t1=0x100010 --defsym t2=0x100020 --defsym t3=0x100030 \
# RUN:   --defsym t4=0x100040 --defsym t5=0x100050 --defsym t6=0x100060 --defsym t7=0x100070 \
# RUN:   --defsym t8=0x100080 --defsym t9=0x100090 --defsym t10=0x1000a0 --defsym t11=0x1000b0 \
# RUN:   --defsym t12=0x1000c0 --defsym t13=0x1000d0 --defsym t14=0x1000e0 --defsym t15=0x1000f0 \
# RUN:   --defsym t16=0x100100 --defsym t17=0x100110 --defsym t18=0x100120 --defsym t19=0x100130 \
# RUN:   --defsym t20=0x100140 --defsym t21=0x100150 --defsym t22=0x100160 --defsym t23=0x100170 \
# RUN:   --defsym t24=0x100180 --defsym t25=0x100190 --defsym t26=0x1001a0 --defsym t27=0x1001b0 \
# RUN:   --defsym t28=0x1001c0 --defsym t29=0x1001d0 --defsym t30=0x1001e0 --defsym t31=0x1001f0 \
# RUN:   -o %t.rv64

# RUN: llvm-readelf -S %t.rv32 | FileCheck --check-prefix=SEC32 %s
# RUN: llvm-readelf -S %t.rv64 | FileCheck --check-prefix=SEC64 %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=DISASM %s

## 32 entries * 4 bytes = 0x80; 32 entries * 8 bytes = 0x100.
# SEC32: .riscv.jvt PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000080
# SEC64: .riscv.jvt PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000100

## Verify some instructions were converted.
# DISASM: cm.jt

.global _start
.p2align 3
_start:
## Use enough repetitions per target so that the savings (2 bytes per tail on
## RV64 after jal relaxation) exceed the table entry cost (8 bytes on RV64).
.altmacro
.macro iota n, i=0
.if \n-\i
  .rept 6
  tail t\i
  .endr
  iota \n, %(\i+1)
.endif
.endm

iota 32
