# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc --filetype=obj --triple=riscv32-unknown-linux-gnu rv32-func1-zicfiss.s -o rv32-func1-zicfiss.o
# RUN: llvm-mc --filetype=obj --triple=riscv32-unknown-linux-gnu func2.s -o rv32-func2.o
# RUN: llvm-mc --filetype=obj --triple=riscv32-unknown-linux-gnu rv32-func2-zicfiss.s -o rv32-func2-zicfiss.o
# RUN: llvm-mc --filetype=obj --triple=riscv32-unknown-linux-gnu func3.s -o rv32-func3.o
# RUN: llvm-mc --filetype=obj --triple=riscv32-unknown-linux-gnu rv32-func3-zicfiss.s -o rv32-func3-zicfiss.o
# RUN: llvm-mc --filetype=obj --triple=riscv64-unknown-linux-gnu rv64-func1-zicfiss.s -o rv64-func1-zicfiss.o
# RUN: llvm-mc --filetype=obj --triple=riscv64-unknown-linux-gnu func2.s -o rv64-func2.o
# RUN: llvm-mc --filetype=obj --triple=riscv64-unknown-linux-gnu rv64-func2-zicfiss.s -o rv64-func2-zicfiss.o
# RUN: llvm-mc --filetype=obj --triple=riscv64-unknown-linux-gnu func3.s -o rv64-func3.o
# RUN: llvm-mc --filetype=obj --triple=riscv64-unknown-linux-gnu rv64-func3-zicfiss.s -o rv64-func3-zicfiss.o

## ZICFISS should be enabled when it's enabled in all inputs
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func2-zicfiss.o rv32-func3-zicfiss.o -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFISS %s
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func3-zicfiss.o --shared -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFISS %s
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func2-zicfiss.o rv64-func3-zicfiss.o -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFISS %s
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func3-zicfiss.o --shared -o - | llvm-readelf -n - | FileCheck --check-prefix ZICFISS %s
# ZICFISS: Properties: RISC-V feature: ZICFISS

## ZICFISS should not be enabled if it's not enabled in at least one input.
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func2.o rv32-func3-zicfiss.o -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv32-func2-zicfiss.o rv32-func3.o --shared -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func2.o rv64-func3-zicfiss.o -o - | llvm-readelf -n - | count 0
# RUN: ld.lld rv64-func2-zicfiss.o rv64-func3.o --shared -o - | llvm-readelf -n - | count 0

## zicfiss-report should report any input files that don't have the zicfiss property.
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func2.o rv32-func3-zicfiss.o -o /dev/null -z zicfiss-report=warning 2>&1 | FileCheck --check-prefix=MISS-SS-WARN %s
# RUN: not ld.lld rv32-func2-zicfiss.o rv32-func3.o --shared -o /dev/null -z zicfiss-report=error 2>&1 | FileCheck --check-prefix=MISS-SS-ERROR %s
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func2-zicfiss.o rv32-func3-zicfiss.o -o /dev/null -z zicfiss-report=warning 2>&1 | count 0
# RUN: ld.lld rv32-func1-zicfiss.o rv32-func2-zicfiss.o rv32-func3-zicfiss.o -o /dev/null -z zicfiss-report=error 2>&1 | count 0
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func2.o rv64-func3-zicfiss.o -o /dev/null -z zicfiss-report=warning 2>&1 | FileCheck --check-prefix=MISS-SS-WARN %s
# RUN: not ld.lld rv64-func2-zicfiss.o rv64-func3.o --shared -o /dev/null -z zicfiss-report=error 2>&1 | FileCheck --check-prefix=MISS-SS-ERROR %s
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func2-zicfiss.o rv64-func3-zicfiss.o -o /dev/null -z zicfiss-report=warning 2>&1 | count 0
# RUN: ld.lld rv64-func1-zicfiss.o rv64-func2-zicfiss.o rv64-func3-zicfiss.o -o /dev/null -z zicfiss-report=error 2>&1 | count 0
# MISS-SS-WARN: warning: rv{{32|64}}-func2.o: -z zicfiss-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS property
# MISS-SS-ERROR: error: rv{{32|64}}-func3.o: -z zicfiss-report: file does not have GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS property

## An invalid zicfiss option should give an error
# RUN: not ld.lld rv32-func1-zicfiss.o rv32-func2-zicfiss.o rv32-func3-zicfiss.o -z zicfiss-report=nonsense 2>&1 | FileCheck --check-prefix=INVALID-REPORT %s
# RUN: not ld.lld rv64-func1-zicfiss.o rv64-func2-zicfiss.o rv64-func3-zicfiss.o -z zicfiss-report=nonsense 2>&1 | FileCheck --check-prefix=INVALID-REPORT %s
# INVALID-REPORT: error: -z zicfiss-report= parameter nonsense is not recognized

#--- rv32-func1-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
.balign 4
ndesc_end:

.text
.globl _start
.type func1,%function
func1:
  call func2
  ret

#--- rv64-func1-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
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

#--- rv32-func2-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
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

#--- rv64-func2-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
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

#--- rv32-func3-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
.balign 4
ndesc_end:

.text
.globl func3
.type func3,@function
func3:
  ret

#--- rv64-func3-zicfiss.s

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
.4byte 2          // GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS
.balign 8
ndesc_end:

.text
.globl func3
.type func3,@function
func3:
  ret
