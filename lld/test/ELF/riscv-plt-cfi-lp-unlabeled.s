# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 rv32-foo.s -o foo32.o
# RUN: ld.lld -shared foo32.o -soname=libfoo32.so -z zicfilp-unlabeled-report=error --fatal-warnings -o libfoo32.so
# RUN: llvm-mc -filetype=obj -triple=riscv32 rv32-start.s -o start32.o
# RUN: ld.lld start32.o libfoo32.so -z zicfilp-unlabeled-report=error --fatal-warnings -o out32
# RUN: llvm-readelf -S out32 | FileCheck --check-prefix=SEC32 %s
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+experimental-zicfilp out32 | FileCheck --check-prefixes=DIS,DIS32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 rv64-foo.s -o foo64.o
# RUN: ld.lld -shared foo64.o -soname=libfoo64.so -z zicfilp-unlabeled-report=error --fatal-warnings -o libfoo64.so
# RUN: llvm-mc -filetype=obj -triple=riscv64 rv64-start.s -o start64.o
# RUN: ld.lld start64.o libfoo64.so -z zicfilp-unlabeled-report=error --fatal-warnings -o out64
# RUN: llvm-readelf -S out64 | FileCheck --check-prefix=SEC64 %s
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+experimental-zicfilp out64 | FileCheck --check-prefixes=DIS,DIS64 %s

# SEC32: .plt     PROGBITS {{0*}}00011210
# SEC32: .got.plt PROGBITS {{0*}}000132b8

# SEC64: .plt     PROGBITS {{0*}}00011330
# SEC64: .got.plt PROGBITS {{0*}}00013440

# DIS:      Disassembly of section .plt:
# DIS:      <.plt>:
# DIS-NEXT:     lpad 0x0
# DIS-NEXT:     auipc t2, 0x2
# DIS-NEXT:     sub t1, t1, t3
# DIS32-NEXT:   lw t3, 0xa4(t2)
# DIS64-NEXT:   ld t3, 0x10c(t2)
# DIS-NEXT:     addi t1, t1, -0x40
# DIS32-NEXT:   addi t0, t2, 0xa4
# DIS64-NEXT:   addi t0, t2, 0x10c
# DIS32-NEXT:   srli t1, t1, 0x2
# DIS64-NEXT:   srli t1, t1, 0x1
# DIS32-NEXT:   lw t0, 0x4(t0)
# DIS64-NEXT:   ld t0, 0x8(t0)
# DIS-NEXT:     jr t3
# DIS-NEXT:     nop
# DIS-NEXT:     nop
# DIS-NEXT:     nop

# DIS:          lpad 0x0
# DIS-NEXT:     auipc t3, 0x2
# DIS32-NEXT:   lw t3, 0x7c(t3)
# DIS64-NEXT:   ld t3, 0xec(t3)
# DIS-NEXT:     jalr t1, t3

#--- rv32-start.s

.section ".note.gnu.property", "a"
.balign 4
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 4
.4byte 0xc0000000 // GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 4
ndesc_end:

.text
.global _start, foo

_start:
  call foo@plt

#--- rv32-foo.s

.section ".note.gnu.property", "a"
.balign 4
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 4
.4byte 0xc0000000 // GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 4
ndesc_end:

.text
.global foo
.type foo, @function
foo:
  ret

#--- rv64-start.s

.section ".note.gnu.property", "a"
.balign 8
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 8
.4byte 0xc0000000 // GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 8
ndesc_end:

.text
.global _start, foo

_start:
  call foo@plt

#--- rv64-foo.s

.section ".note.gnu.property", "a"
.balign 8
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 8
.4byte 0xc0000000 // GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 8
ndesc_end:

.text
.global foo
.type foo, @function
foo:
  ret
