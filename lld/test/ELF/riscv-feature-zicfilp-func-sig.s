# REQUIRES: riscv
## Test the ZICFILP func-sig feature.
## To lift maintenance burden, most tests are conducted only with 64-bit RISC-V
## Naming convention: *-s.s files enables ZICFILP func-sig.
## Naming convention: *-u.s files enables ZICFILP unlabeled.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f1-s.s -o rv32-f1-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f2-s.s -o rv32-f2-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-f3-s.s -o rv32-f3-s.o

# RUN: llvm-mc --filetype=obj --triple=riscv64 f1-s.s -o f1-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f2.s   -o f2.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f2-s.s -o f2-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3.s   -o f3.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3-s.s -o f3-s.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 f3-u.s -o f3-u.o

## ZICFILP-func-sig should be enabled when it's enabled in all inputs or when
## it's forced on.
# RUN: ld.lld rv32-f1-s.o rv32-f2-s.o rv32-f3-s.o -o out.rv32 --fatal-warnings
# RUN: llvm-readelf -n out.rv32 | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -o out --fatal-warnings
# RUN: llvm-readelf -n out | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f3-s.o --shared -o out.so --fatal-warnings
# RUN: llvm-readelf -n out.so | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.force -z zicfilp=func-sig --fatal-warnings
# RUN: llvm-readelf -n out.force | FileCheck --check-prefix=ZICFILP %s
# RUN: ld.lld f2-s.o f3.o --shared -o out.force.so -z zicfilp=never -z zicfilp=func-sig --fatal-warnings
# RUN: llvm-readelf -n out.force.so | FileCheck --check-prefix=ZICFILP %s
# ZICFILP: Properties: RISC-V feature: ZICFILP-func-sig

## ZICFILP-func-sig should not be enabled if it's not enabled in at least one
## input
# RUN: ld.lld f1-s.o f2.o f3-s.o -o out.no --fatal-warnings
# RUN: llvm-readelf -n out.no | count 0
# RUN: ld.lld f2-s.o f3.o --shared -o out.no.so --fatal-warnings
# RUN: llvm-readelf -n out.no.so | count 0

## ZICFILP-func-sig should be disabled with zicfilp=never, even if
## ZICFILP-func-sig is present in all inputs.
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z zicfilp=func-sig -z zicfilp=never -o out.never --fatal-warnings
# RUN: llvm-readelf -n out.never | count 0

## zicfilp-func-sig-report should report any input files that don't have the
## ZICFILP-func-sig property
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-func-sig-report=warning 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-func-sig-report=warning -z zicfilp=func-sig 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: ld.lld f1-s.o f2.o f3-s.o -z zicfilp-func-sig-report=warning -z zicfilp=never 2>&1 | FileCheck --check-prefix=REPORT-WARN %s
# RUN: not ld.lld f2-s.o f3.o --shared -z zicfilp-func-sig-report=error 2>&1 | FileCheck --check-prefix=REPORT-ERROR %s
# RUN: ld.lld f1-s.o f2-s.o f3-s.o -z zicfilp-func-sig-report=warning -z zicfilp=func-sig 2>&1 | count 0
# REPORT-WARN: warning: f2.o: -z zicfilp-func-sig-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG property
# REPORT-ERROR: error: f3.o: -z zicfilp-func-sig-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG property

## An invalid -z zicfilp-func-sig-report option should give an error
# RUN: not ld.lld f2-s.o -z zicfilp-func-sig-report=x 2>&1 | FileCheck --check-prefix=INVALID %s
# INVALID: error: unknown -z zicfilp-func-sig-report= value: x

## ZICFILP-unlabeled and ZICFILP-func-sig should conflict with each other.
# RUN: ld.lld f3-u.o -o out.override -z zicfilp=func-sig 2>&1 | FileCheck --check-prefix=FORCE-CONFLICT %s
# FORCE-CONFLICT: warning: f3-u.o: -z zicfilp=func-sig: file has conflicting property: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_UNLABELED

## -z zicfilp=func-sig should override and disable ZICFILP-unlabeled.
# RUN: llvm-readelf -n out.override | FileCheck --check-prefixes=ZICFILP,OVERRIDE %s
# OVERRIDE-NOT: ZICFILP-unlabeled

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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
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
.4byte 4          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG
.balign 8
ndesc_end:

.text
.globl f3
.type f3,@function
f3:
  ret

#--- f3-u.s
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
