# REQUIRES: systemz

# RUN: llvm-mc --filetype=obj --triple=s390x-unknown-linux -mcpu=z13 %s -o %t.o

# RUN: ld.lld %t.o --defsym foo16=pc16dbl+4 --defsym bar16=pc16dbl --defsym foo32=pc32dbl+6 --defsym bar32=pc32dbl --defsym foo12=pc12dbl+6 --defsym bar12=pc12dbl --defsym foo24=pc24dbl+6 --defsym bar24=pc24dbl -o %t
# RUN: llvm-objdump --no-show-raw-insn --mcpu=z13 -d %t | FileCheck %s --check-prefix=CHECK
# CHECK: 0000000001001120 <pc16dbl>:
# CHECK: je      0x1001124
# CHECK: jne     0x1001120
# CHECK: 0000000001001128 <pc32dbl>:
# CHECK: jge     0x100112e
# CHECK: jgne    0x1001128
# CHECK: 0000000001001134 <pc12dbl>:
# CHECK: bprp    5, 0x100113a, 0x1001134
# CHECK: bprp    6, 0x1001134, 0x100113a
# CHECK: 0000000001001140 <pc24dbl>:
# CHECK: bprp    5, 0x1001140, 0x1001146
# CHECK: bprp    6, 0x1001146, 0x1001140

# RUN: ld.lld %t.o --defsym foo16=pc16dbl+0xfffe --defsym bar16=pc16dbl+4-0x10000 --defsym foo32=pc32dbl+0xfffffffe --defsym bar32=pc32dbl+6-0x100000000 --defsym foo12=pc12dbl+0xffe --defsym bar12=pc12dbl+6-0x1000 --defsym foo24=pc24dbl+0xfffffe --defsym bar24=pc24dbl+6-0x1000000 -o %t.limits
# RUN: llvm-objdump --no-show-raw-insn --mcpu=z13 -d %t.limits | FileCheck %s --check-prefix=LIMITS
# LIMITS:      je   0x101111e
# LIMITS-NEXT: jne  0xff1124
# LIMITS:      jge  0x101001126
# LIMITS-NEXT: jgne 0xffffffff0100112e
# LIMITS:      bprp 5, 0x1002132, 0x1001134
# LIMITS-NEXT: bprp 6, 0x100013a, 0x100113a
# LIMITS:      bprp 5, 0x1001140, 0x200113e
# LIMITS-NEXT: bprp 6, 0x1001146, 0x1146

# RUN: not ld.lld %t.o --defsym foo16=pc16dbl+0x10000 --defsym bar16=pc16dbl+4-0x10002 --defsym foo32=pc32dbl+0x100000000 --defsym bar32=pc32dbl+6-0x100000002 --defsym foo12=pc12dbl+0x1000 --defsym bar12=pc12dbl+6-0x1002 --defsym foo24=pc24dbl+0x1000000 --defsym bar24=pc24dbl+6-0x1000002 -o /dev/null 2>&1 | FileCheck -DFILE=%t.o --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: error: [[FILE]]:(.text+0x2): relocation R_390_PC16DBL out of range: 65536 is not in [-65536, 65535]; references 'foo16'
# ERROR-RANGE: error: [[FILE]]:(.text+0x6): relocation R_390_PC16DBL out of range: -65538 is not in [-65536, 65535]; references 'bar16'
# ERROR-RANGE: error: [[FILE]]:(.text+0xa): relocation R_390_PC32DBL out of range: 4294967296 is not in [-4294967296, 4294967295]; references 'foo32'
# ERROR-RANGE: error: [[FILE]]:(.text+0x10): relocation R_390_PC32DBL out of range: -4294967298 is not in [-4294967296, 4294967295]; references 'bar32'
# ERROR-RANGE: error: [[FILE]]:(.text+0x15): relocation R_390_PC12DBL out of range: 4096 is not in [-4096, 4095]; references 'foo12'
# ERROR-RANGE: error: [[FILE]]:(.text+0x1b): relocation R_390_PC12DBL out of range: -4098 is not in [-4096, 4095]; references 'bar12'
# ERROR-RANGE: error: [[FILE]]:(.text+0x23): relocation R_390_PC24DBL out of range: 16777216 is not in [-16777216, 16777215]; references 'foo24'
# ERROR-RANGE: error: [[FILE]]:(.text+0x29): relocation R_390_PC24DBL out of range: -16777218 is not in [-16777216, 16777215]; references 'bar24'

# RUN: not ld.lld %t.o --defsym foo16=pc16dbl+1 --defsym bar16=pc16dbl-1 --defsym foo32=pc32dbl+1 --defsym bar32=pc32dbl-1 --defsym foo12=pc12dbl+1 --defsym bar12=pc12dbl-1 --defsym foo24=pc24dbl+1 --defsym bar24=pc24dbl-1 -o /dev/null 2>&1 | FileCheck -DFILE=%t.o --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN:      error: [[FILE]]:(.text+0x2): improper alignment for relocation R_390_PC16DBL: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x6): improper alignment for relocation R_390_PC16DBL: 0xFFFFFFFFFFFFFFFB is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0xa): improper alignment for relocation R_390_PC32DBL: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x10): improper alignment for relocation R_390_PC32DBL: 0xFFFFFFFFFFFFFFF9 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x15): improper alignment for relocation R_390_PC12DBL: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x1b): improper alignment for relocation R_390_PC12DBL: 0xFFFFFFFFFFFFFFF9 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x23): improper alignment for relocation R_390_PC24DBL: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN-NEXT: error: [[FILE]]:(.text+0x29): improper alignment for relocation R_390_PC24DBL: 0xFFFFFFFFFFFFFFF9 is not aligned to 2 bytes

.global _start
.global pc16dbl
.global pc32dbl
.global pc12dbl
.global pc24dbl
_start:
pc16dbl:
     je   foo16
     jne  bar16
pc32dbl:
     jge  foo32
     jgne bar32
pc12dbl:
     bprp 5,foo12,0
     bprp 6,bar12,0
pc24dbl:
     bprp 5,0,foo24
     bprp 6,0,bar24
