# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-func1-zicfilp.s -o rv32-func1-zicfilp.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 func2.s              -o rv32-func2.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-func2-zicfilp.s -o rv32-func2-zicfilp.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 func3.s              -o rv32-func3.o
# RUN: llvm-mc --filetype=obj --triple=riscv32 rv32-func3-zicfilp.s -o rv32-func3-zicfilp.o

# RUN: llvm-mc --filetype=obj --triple=riscv64 rv64-func1-zicfilp.s -o rv64-func1-zicfilp.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 func2.s              -o rv64-func2.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 rv64-func2-zicfilp.s -o rv64-func2-zicfilp.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 func3.s              -o rv64-func3.o
# RUN: llvm-mc --filetype=obj --triple=riscv64 rv64-func3-zicfilp.s -o rv64-func3-zicfilp.o

## ZICFILP-func-sig should be enabled when it's enabled in all inputs
# RUN: ld.lld rv32-func1-zicfilp.o rv32-func2-zicfilp.o rv32-func3-zicfilp.o   \
# RUN:  -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFILP %s
# RUN: ld.lld rv32-func1-zicfilp.o rv32-func3-zicfilp.o --shared               \
# RUN:  -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFILP %s
# RUN: ld.lld rv64-func1-zicfilp.o rv64-func2-zicfilp.o rv64-func3-zicfilp.o   \
# RUN:  -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFILP %s
# RUN: ld.lld rv64-func1-zicfilp.o rv64-func3-zicfilp.o --shared               \
# RUN:  -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFILP %s
# ZICFILP: Properties: RISC-V feature: ZICFILP-func-sig

## ZICFILP-func-sig should not be enabled if it's not enabled in at least one
## input
# RUN: ld.lld rv32-func1-zicfilp.o rv32-func2.o rv32-func3-zicfilp.o           \
# RUN:  -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv32-func2-zicfilp.o rv32-func3.o --shared                       \
# RUN:  -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv64-func1-zicfilp.o rv64-func2.o rv64-func3-zicfilp.o           \
# RUN:  -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv64-func2-zicfilp.o rv64-func3.o --shared                       \
# RUN:  -o - | llvm-readelf -n - | count 0

## zicfilp-func-sig-report should report any input files that don't have the
## zicfilp-func-sig property
# RUN: ld.lld rv32-func1-zicfilp.o rv32-func2.o rv32-func3-zicfilp.o           \
# RUN:  -z zicfilp-func-sig-report=warning 2>&1                                \
# RUN:  | FileCheck --check-prefix=MISS-LP-WARN %s
# RUN: not ld.lld rv32-func2-zicfilp.o rv32-func3.o --shared                   \
# RUN:  -z zicfilp-func-sig-report=error   2>&1                                \
# RUN:  | FileCheck --check-prefix=MISS-LP-ERROR %s

# RUN: ld.lld rv32-func1-zicfilp.o rv32-func2-zicfilp.o rv32-func3-zicfilp.o   \
# RUN:  -z zicfilp-func-sig-report=warning 2>&1 | count 0
# RUN: ld.lld rv32-func1-zicfilp.o rv32-func2-zicfilp.o rv32-func3-zicfilp.o   \
# RUN:  -z zicfilp-func-sig-report=error   2>&1 | count 0

# RUN: ld.lld rv64-func1-zicfilp.o rv64-func2.o rv64-func3-zicfilp.o           \
# RUN:  -z zicfilp-func-sig-report=warning 2>&1                                \
# RUN:  | FileCheck --check-prefix=MISS-LP-WARN %s
# RUN: not ld.lld rv64-func2-zicfilp.o rv64-func3.o --shared                   \
# RUN:  -z zicfilp-func-sig-report=error   2>&1                                \
# RUN:  | FileCheck --check-prefix=MISS-LP-ERROR %s

# RUN: ld.lld rv64-func1-zicfilp.o rv64-func2-zicfilp.o rv64-func3-zicfilp.o   \
# RUN:  -z zicfilp-func-sig-report=warning 2>&1 | count 0
# RUN: ld.lld rv64-func1-zicfilp.o rv64-func2-zicfilp.o rv64-func3-zicfilp.o   \
# RUN:  -z zicfilp-func-sig-report=error   2>&1 | count 0

# MISS-LP-WARN: warning: rv{{32|64}}-func2.o: -z zicfilp-func-sig-report:
# MISS-LP-WARN-SAME: file does not have
# MISS-LP-WARN-SAME: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG property
# MISS-LP-ERROR: error: rv{{32|64}}-func3.o: -z zicfilp-func-sig-report:
# MISS-LP-ERROR-SAME: file does not have
# MISS-LP-ERROR-SAME: GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_FUNC_SIG property

## An invalid -z zicfilp-func-sig-report option should give an error
# RUN: not ld.lld rv32-func2-zicfilp.o rv32-func3-zicfilp.o                    \
# RUN:  -z zicfilp-func-sig-report=nonsense 2>&1                               \
# RUN:  | FileCheck --check-prefix=INVALID-REPORT %s
# RUN: not ld.lld rv64-func2-zicfilp.o rv64-func3-zicfilp.o                    \
# RUN:  -z zicfilp-func-sig-report=nonsense 2>&1                               \
# RUN:  | FileCheck --check-prefix=INVALID-REPORT %s
# INVALID-REPORT: error: unknown -z zicfilp-func-sig-report= value: nonsense

#--- rv32-func1-zicfilp.s

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
.type func1,%function
func1:
  call func2
  ret

#--- rv64-func1-zicfilp.s

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
.type func1,%function
func1:
  call func2
  ret

#--- func2.s

.text
.globl func2
.type func2,@function
func2:
  .globl func3
  .type func3, @function
  call func3
  ret

#--- rv32-func2-zicfilp.s

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
.globl func2
.type func2,@function
func2:
  .globl func3
  .type func3, @function
  call func3
  ret

#--- rv64-func2-zicfilp.s

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
.globl func2
.type func2,@function
func2:
  .globl func3
  .type func3, @function
  call func3
  ret

#--- func3.s

.text
.globl func3
.type func3,@function
func3:
  ret

#--- rv32-func3-zicfilp.s

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
.globl func3
.type func3,@function
func3:
  ret

#--- rv64-func3-zicfilp.s

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
.globl func3
.type func3,@function
func3:
  ret
