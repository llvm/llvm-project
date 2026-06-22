# REQUIRES: riscv
## Test the ZICFILP unlabeled feature.
## To lift maintenance burden, most tests are conducted only with 64-bit RISC-V
## Naming convention: *-s.s files enables ZICFILP unlabeled.
## Naming convention: *-f.s files enables ZICFILP func-sig.
## Naming convention: *-c.s files enables both of the conflicting ZICFILP unlabeled and ZICFILP func-sig features.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f1-s.s -o rv32-f1-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f2-s.s -o rv32-f2-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f3-s.s -o rv32-f3-s.o

# RUN: llvm-mc --filetype=obj --triple=riscv64 f1-s.s -o f1-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f1-c.s -o f1-c.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f2.s   -o f2.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f2-s.s -o f2-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3.s   -o f3.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3-s.s -o f3-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3-f.s -o f3-f.o

## Test PLT generation with unlabeled landing pads
# RUN: llvm-mc -filetype=obj -triple=riscv32 rv32-plt-foo.s -o rv32-plt-foo.o
# RUN: ld.lld -shared rv32-plt-foo.o -soname=libfoo32.so -z zicfilp-unlabeled-report=error --fatal-warnings -o libfoo32.so
# RUN: llvm-mc -filetype=obj -triple=riscv32 rv32-plt-start.s -o rv32-plt-start.o
# RUN: ld.lld rv32-plt-start.o libfoo32.so -z zicfilp-unlabeled-report=error --fatal-warnings -o out.plt32
# RUN: llvm-readelf -S out.plt32 | FileCheck --check-prefix=PLT-SEC32 %s
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+experimental-zicfilp out.plt32 | FileCheck --check-prefixes=PLT-DIS,PLT-DIS32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 rv64-plt-foo.s -o rv64-plt-foo.o
# RUN: ld.lld -shared rv64-plt-foo.o -soname=libfoo64.so -z zicfilp-unlabeled-report=error --fatal-warnings -o libfoo64.so
# RUN: llvm-mc -filetype=obj -triple=riscv64 rv64-plt-start.s -o rv64-plt-start.o
# RUN: ld.lld rv64-plt-start.o libfoo64.so -z zicfilp-unlabeled-report=error --fatal-warnings -o out.plt64
# RUN: llvm-readelf -S out.plt64 | FileCheck --check-prefix=PLT-SEC64 %s
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+experimental-zicfilp out.plt64 | FileCheck --check-prefixes=PLT-DIS,PLT-DIS64 %s

## ZICFILP-unlabeled should be enabled when it's enabled in all inputs or when
## it's forced on.
# RUN: ld.lld rv32-f1-s.o rv32-f2-s.o rv32-f3-s.o -o out.rv32 --fatal-warnings
# RUN: llvm-readelf -n out.rv32 | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -o out --fatal-warnings
# RUN: llvm-readelf -n out | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f3-s.o --shared -o out.so --fatal-warnings
# RUN: llvm-readelf -n out.so | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.force -z zicfilp=unlabeled --fatal-warnings
# RUN: llvm-readelf -n out.force | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f2-s.o f3.o --shared -o out.force.so -z zicfilp=never -z zicfilp=unlabeled --fatal-warnings
# RUN: llvm-readelf -n out.force.so | FileCheck --check-prefix=ZICFILP %s
# ZICFILP: Properties: RISC-V feature: ZICFILP-unlabeled

## ZICFILP-unlabeled should not be enabled if it's not enabled in at least one
## input
# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.no --fatal-warnings
# RUN: llvm-readelf -n out.no | count 0
# RUN: ld.lld f2-s.o f3.o --shared -o out.no.so --fatal-warnings
# RUN: llvm-readelf -n out.no.so | count 0

## ZICFILP-unlabeled should be disabled with zicfilp=never, even if
## ZICFILP-unlabeled is present in all inputs.
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z zicfilp=unlabeled -z zicfilp=never -o out.never --fatal-warnings
# RUN: llvm-readelf -n out.never | count 0

## zicfilp-unlabeled-report should report any input files that don't have the
## ZICFILP-unlabeled property
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-unlabeled-report=warning 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-unlabeled-report=warning -z zicfilp=unlabeled 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-unlabeled-report=warning -z zicfilp=never 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: not ld.lld f2-s.o f3.o --shared -z zicfilp-unlabeled-report=error 2>&1 | FileCheck --check-prefix=REPORT-ERROR %s
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z zicfilp-unlabeled-report=warning -z zicfilp=never 2>&1 | count 0
# REPORT-WARN: warning: f2.o: -z zicfilp-unlabeled-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED property
# REPORT-ERROR: error: f3.o: -z zicfilp-unlabeled-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED property

## An invalid -z zicfilp-unlabeled-report option should give an error
# RUN: not ld.lld f2-s.o -z zicfilp=x -z zicfilp-unlabeled-report=x 2>&1 | FileCheck --check-prefix=INVALID %s
# INVALID: error: unknown -z zicfilp= value: x
# INVALID: error: unknown -z zicfilp-unlabeled-report= value: x

## ZICFILP-unlabeled and ZICFILP-func-sig should conflict with each other
# RUN: not ld.lld f1-c.o 2>&1 | FileCheck --check-prefix=CONFLICT %s
# RUN: ld.lld f3-f.o -o out.override -z zicfilp=unlabeled 2>&1 | FileCheck --check-prefix=FORCE-CONFLICT %s
# CONFLICT: error: f1-c.o: file has conflicting properties: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED and GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
# FORCE-CONFLICT: warning: f3-f.o: -z zicfilp=unlabeled: file has conflicting property: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG

## -z zicfilp=unlabeled should override and disable ZICFILP-func-sig.
# RUN: llvm-readelf -n out.override | FileCheck --check-prefixes=ZICFILP,OVERRIDE %s
# OVERRIDE-NOT: ZICFILP-func-sig

# PLT-SEC32: .plt     PROGBITS {{0*}}00011210
# PLT-SEC32: .got.plt PROGBITS {{0*}}000132a8

# PLT-SEC64: .plt     PROGBITS {{0*}}00011330
# PLT-SEC64: .got.plt PROGBITS {{0*}}00013430

# PLT-DIS:      Disassembly of section .plt:
# PLT-DIS:      <.plt>:
# PLT-DIS-NEXT:     auipc t3, 0x2
# PLT-DIS-NEXT:     sub t1, t1, t2
# PLT-DIS32-NEXT:   lw t2, 0x98(t3)
# PLT-DIS64-NEXT:   ld t2, 0x100(t3)
# PLT-DIS-NEXT:     addi t1, t1, -0x30
# PLT-DIS32-NEXT:   addi t0, t3, 0x98
# PLT-DIS64-NEXT:   addi t0, t3, 0x100
# PLT-DIS32-NEXT:   srli t1, t1, 0x2
# PLT-DIS64-NEXT:   srli t1, t1, 0x1
# PLT-DIS32-NEXT:   lw t0, 0x4(t0)
# PLT-DIS64-NEXT:   ld t0, 0x8(t0)
# PLT-DIS-NEXT:     jr t2

# PLT-DIS:          lpad 0x0
# PLT-DIS-NEXT:     auipc t2, 0x2
# PLT-DIS32-NEXT:   lw t2, 0x7c(t2)
# PLT-DIS64-NEXT:   ld t2, 0xec(t2)
# PLT-DIS-NEXT:     jalr t1, t2

#--- rv32-f1-s.s
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
.globl _start
.type f1,%function
f1:
  call f2
  ret

#--- f1-s.s
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
.globl _start
.type f1,%function
f1:
  call f2
  ret

#--- f1-c.s
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
.4byte 5          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED | GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
.balign 8
ndesc_end:

.text
.globl _start
.type f1,%function
f1:
  call f2
  ret

#--- f2.s
.text
.globl f2
.type f2,@function
f2:
  .globl f3
  .type f3, @function
  call f3
  ret

#--- rv32-f2-s.s
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
.globl f2
.type f2,@function
f2:
  .globl f3
  .type f3, @function
  call f3
  ret

#--- f2-s.s
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
.globl f2
.type f2,@function
f2:
  .globl f3
  .type f3, @function
  call f3
  ret

#--- f3.s
.text
.globl f3
.type f3,@function
f3:
  ret

#--- rv32-f3-s.s
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
.globl f3
.type f3,@function
f3:
  ret

#--- f3-s.s
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
.globl f3
.type f3,@function
f3:
  ret

#--- f3-f.s
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
.balign 8
ndesc_end:

.text
.globl f3
.type f3,@function
f3:
  ret

#--- rv32-plt-start.s
.section ".note.gnu.property", "a"
.balign 4
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        # NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 4
.4byte 0xc0000000 # GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          # GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 4
ndesc_end:

.text
.global _start, foo

_start:
  call foo@plt

#--- rv32-plt-foo.s
.section ".note.gnu.property", "a"
.balign 4
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        # NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 4
.4byte 0xc0000000 # GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          # GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 4
ndesc_end:

.text
.global foo
.type foo, @function
foo:
  ret

#--- rv64-plt-start.s
.section ".note.gnu.property", "a"
.balign 8
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        # NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 8
.4byte 0xc0000000 # GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          # GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 8
ndesc_end:

.text
.global _start, foo

_start:
  call foo@plt

#--- rv64-plt-foo.s
.section ".note.gnu.property", "a"
.balign 8
.4byte 4
.4byte (ndesc_end - ndesc_begin)
.4byte 0x5        # NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"
ndesc_begin:
.balign 8
.4byte 0xc0000000 # GNU_PROPERTY_RISCV_FEATURE_1_AND
.4byte 4
.4byte 1          # GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED
.balign 8
ndesc_end:

.text
.global foo
.type foo, @function
foo:
  ret
