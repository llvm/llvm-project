# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %s -o %t.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.la64.o

# RUN: ld.lld %t.la32.o --defsym foo16=b16+4 --defsym bar16=b16 --defsym foo21=b21+4 --defsym bar21=b21 --defsym foo26=b26+4 --defsym bar26=b26 -o %t.la32
# RUN: ld.lld %t.la64.o --defsym foo16=b16+4 --defsym bar16=b16 --defsym foo21=b21+4 --defsym bar21=b21 --defsym foo26=b26+4 --defsym bar26=b26 -o %t.la64
# RUN: llvm-objdump --no-show-raw-insn -d %t.la32 | FileCheck %s --check-prefix=CHECK
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64 | FileCheck %s --check-prefix=CHECK
# CHECK: beq $zero, $zero, 4
# CHECK: bne $zero, $zero, -4
# CHECK: beqz $s8, 4
# CHECK: bnez $s8, -4
# CHECK: b 4
# CHECK: bl -4

# RUN: ld.lld %t.la32.o --defsym foo16=b16+0x1fffc --defsym bar16=b16+4-0x20000 --defsym foo21=b21+0x3ffffc --defsym bar21=b21+4-0x400000 --defsym foo26=b26+0x7fffffc --defsym bar26=b26+4-0x8000000 -o %t.la32.limits
# RUN: ld.lld %t.la64.o --defsym foo16=b16+0x1fffc --defsym bar16=b16+4-0x20000 --defsym foo21=b21+0x3ffffc --defsym bar21=b21+4-0x400000 --defsym foo26=b26+0x7fffffc --defsym bar26=b26+4-0x8000000 -o %t.la64.limits
# RUN: llvm-objdump --no-show-raw-insn -d %t.la32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump --no-show-raw-insn -d %t.la64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      beq $zero, $zero, 131068
# LIMITS-NEXT: bne $zero, $zero, -131072
# LIMITS:      beqz $s8, 4194300
# LIMITS-NEXT: bnez $s8, -4194304
# LIMITS:      b 134217724
# LIMITS-NEXT: bl -134217728

# RUN: not ld.lld %t.la32.o --defsym foo16=b16+0x20000 --defsym bar16=b16+4-0x20004 --defsym foo21=b21+0x400000 --defsym bar21=b21+4-0x400004 --defsym foo26=b26+0x8000000 --defsym bar26=b26+4-0x8000004 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la32.o --check-prefix=ERROR-RANGE %s
# RUN: not ld.lld %t.la64.o --defsym foo16=b16+0x20000 --defsym bar16=b16+4-0x20004 --defsym foo21=b21+0x400000 --defsym bar21=b21+4-0x400004 --defsym foo26=b26+0x8000000 --defsym bar26=b26+4-0x8000004 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la64.o --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: error: [[FILE]]:(.text+0x0): relocation R_LARCH_B16 out of range: 131072 is not in [-131072, 131071]; references 'foo16'
# ERROR-RANGE: error: [[FILE]]:(.text+0x4): relocation R_LARCH_B16 out of range: -131076 is not in [-131072, 131071]; references 'bar16'
# ERROR-RANGE: error: [[FILE]]:(.text+0x8): relocation R_LARCH_B21 out of range: 4194304 is not in [-4194304, 4194303]; references 'foo21'
# ERROR-RANGE: error: [[FILE]]:(.text+0xc): relocation R_LARCH_B21 out of range: -4194308 is not in [-4194304, 4194303]; references 'bar21'
# ERROR-RANGE: error: [[FILE]]:(.text+0x10): relocation R_LARCH_B26 out of range: 134217728 is not in [-134217728, 134217727]; references 'foo26'
# ERROR-RANGE: error: [[FILE]]:(.text+0x14): relocation R_LARCH_B26 out of range: -134217732 is not in [-134217728, 134217727]; references 'bar26'

# RUN: not ld.lld %t.la32.o --defsym foo16=b16+1 --defsym bar16=b16-1 --defsym foo21=b21+1 --defsym bar21=b21-1 --defsym foo26=b26+1 --defsym bar26=b26-1 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la32.o --check-prefix=ERROR-ALIGN-1 %s
# RUN: not ld.lld %t.la64.o --defsym foo16=b16+1 --defsym bar16=b16-1 --defsym foo21=b21+1 --defsym bar21=b21-1 --defsym foo26=b26+1 --defsym bar26=b26-1 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la64.o --check-prefix=ERROR-ALIGN-1 %s
# ERROR-ALIGN-1:      error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_B16: 0x1 is not aligned to 4 bytes
# ERROR-ALIGN-1-NEXT: error: [[FILE]]:(.text+0x4): improper alignment for relocation R_LARCH_B16: 0xFFFFFFFFFFFFFFFB is not aligned to 4 bytes
# ERROR-ALIGN-1-NEXT: error: [[FILE]]:(.text+0x8): improper alignment for relocation R_LARCH_B21: 0x1 is not aligned to 4 bytes
# ERROR-ALIGN-1-NEXT: error: [[FILE]]:(.text+0xc): improper alignment for relocation R_LARCH_B21: 0xFFFFFFFFFFFFFFFB is not aligned to 4 bytes
# ERROR-ALIGN-1-NEXT: error: [[FILE]]:(.text+0x10): improper alignment for relocation R_LARCH_B26: 0x1 is not aligned to 4 bytes
# ERROR-ALIGN-1-NEXT: error: [[FILE]]:(.text+0x14): improper alignment for relocation R_LARCH_B26: 0xFFFFFFFFFFFFFFFB is not aligned to 4 bytes

# RUN: not ld.lld %t.la32.o --defsym foo16=b16+2 --defsym bar16=b16-2 --defsym foo21=b21+2 --defsym bar21=b21-2 --defsym foo26=b26+2 --defsym bar26=b26-2 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la32.o --check-prefix=ERROR-ALIGN-2 %s
# RUN: not ld.lld %t.la64.o --defsym foo16=b16+2 --defsym bar16=b16-2 --defsym foo21=b21+2 --defsym bar21=b21-2 --defsym foo26=b26+2 --defsym bar26=b26-2 -o /dev/null 2>&1 | FileCheck -DFILE=%t.la64.o --check-prefix=ERROR-ALIGN-2 %s
# ERROR-ALIGN-2:      error: [[FILE]]:(.text+0x0): improper alignment for relocation R_LARCH_B16: 0x2 is not aligned to 4 bytes
# ERROR-ALIGN-2-NEXT: error: [[FILE]]:(.text+0x4): improper alignment for relocation R_LARCH_B16: 0xFFFFFFFFFFFFFFFA is not aligned to 4 bytes
# ERROR-ALIGN-2-NEXT: error: [[FILE]]:(.text+0x8): improper alignment for relocation R_LARCH_B21: 0x2 is not aligned to 4 bytes
# ERROR-ALIGN-2-NEXT: error: [[FILE]]:(.text+0xc): improper alignment for relocation R_LARCH_B21: 0xFFFFFFFFFFFFFFFA is not aligned to 4 bytes
# ERROR-ALIGN-2-NEXT: error: [[FILE]]:(.text+0x10): improper alignment for relocation R_LARCH_B26: 0x2 is not aligned to 4 bytes
# ERROR-ALIGN-2-NEXT: error: [[FILE]]:(.text+0x14): improper alignment for relocation R_LARCH_B26: 0xFFFFFFFFFFFFFFFA is not aligned to 4 bytes

.global _start
.global b16
.global b21
.global b26
_start:
b16:
     beq  $zero, $zero, foo16
     bne  $zero, $zero, bar16
b21:
     beqz $s8, foo21
     bnez $s8, bar21
b26:
     b    foo26
     bl   bar26
