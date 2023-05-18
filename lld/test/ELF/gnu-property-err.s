# REQUIRES: aarch64
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/1.s -o %t1.o
# RUN: not ld.lld %t1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR1

# ERR1: error: {{.*}}.o:(.note.gnu.property+0x0): data is too short

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/2.s -o %t2.o
# RUN: not ld.lld %t2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR2
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/2.s -o %t2be.o
# RUN: not ld.lld %t2be.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR2

# ERR2: error: {{.*}}.o:(.note.gnu.property+0x10): program property is too short

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/3.s -o %t3.o
# RUN: not ld.lld %t3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR3
# RUN: llvm-mc -filetype=obj -triple=aarch64_be %t/3.s -o %t3be.o
# RUN: not ld.lld %t3be.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR3

# ERR3: error: {{.*}}.o:(.note.gnu.property+0x10): FEATURE_1_AND entry is too short

#--- 1.s
.section ".note.gnu.property", "a"
.long 4
.long 17         // n_descsz too long
.long 5          // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4          // pr_datasz
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

#--- 2.s
.section ".note.gnu.property", "a"
.long 4
.long 16         // n_descsz
.long 5          // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 9          // pr_datasz too long
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

#--- 3.s
.section ".note.gnu.property", "a"
.long 4
.long 8          // n_descsz
.long 5          // NT_GNU_PROPERTY_TYPE_0
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 0          // pr_datasz too short
