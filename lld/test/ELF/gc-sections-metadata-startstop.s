# REQUIRES: x86
# LINK_ORDER cnamed sections are not kept alive by the __start_* reference.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --gc-sections -z start-stop-gc -z nostart-stop-gc %t.o -o %t
# RUN: llvm-objdump --section-headers -t %t | FileCheck  %s

## With -z start-stop-gc (default), non-SHF_LINK_ORDER C identifier name
## sections are not retained by __start_/__stop_ references.
# RUN: ld.lld --gc-sections %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck %s --check-prefix=GC
# RUN: ld.lld --gc-sections -z start-stop-gc %t.o -o %t1
# RUN: llvm-readelf -S -s %t1 | FileCheck %s --check-prefix=GC

# CHECK: Sections:
# CHECK-NOT: yy
# CHECK: xx {{.*}} DATA
# CHECK-NOT: yy

# CHECK: SYMBOL TABLE:
# CHECK:   xx    0000000000000000 .protected __start_xx
# CHECK: w *UND* 0000000000000000 __start_yy

# GC:     Section Headers:
# GC-NOT:   xx
# GC-NOT:   yy

# GC:       WEAK DEFAULT UND __start_xx
# GC:       WEAK DEFAULT UND __start_yy

.weak __start_xx
.weak __start_yy

.global _start
_start:
.quad __start_xx
.quad __start_yy

.section xx,"a"
.quad 0

.section .foo,"a"
.quad 0

.section yy,"ao",@progbits,.foo
.quad 0
