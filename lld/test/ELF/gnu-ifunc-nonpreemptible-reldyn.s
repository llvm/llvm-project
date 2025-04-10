# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/a.s -o %t/a.o
# RUN: ld.lld %t/a.o -o %t/a
# RUN: llvm-readelf -S -r %t/a | FileCheck --check-prefix=NOIPLT %s
# RUN: ld.lld %t/a.o -o %t/a.pie -pie
# RUN: llvm-readelf -S -r %t/a.pie | FileCheck --check-prefix=NOIPLT %s
# RUN: ld.lld %t/a.o -o %t/a.android --pack-dyn-relocs=android
# RUN: llvm-readelf -S -r -x .data %t/a.android | FileCheck --check-prefixes=IPLT,IPLT-PDE %s
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/ab
# RUN: llvm-readelf -S -r -x .data %t/ab | FileCheck --check-prefixes=IPLT,IPLT-PDE %s
# RUN: ld.lld %t/a.o %t/b.o -o %t/ab.pie -pie
# RUN: llvm-readelf -S -r %t/ab.pie | FileCheck --check-prefixes=IPLT,IPLT-PIE %s

# NOIPLT-NOT: .iplt
# NOIPLT: .data PROGBITS [[DATA:[0-9a-f]+]]
# NOIPLT: [[DATA]] {{.*}} R_AARCH64_IRELATIVE

# IPLT-PDE: .iplt PROGBITS 0000000000210180
# IPLT-PIE: .iplt
# IPLT-PIE: .data PROGBITS [[DATA:[0-9a-f]+]]
# IPLT: .got.plt PROGBITS [[GOTPLT:[0-9a-f]+]]
# IPLT-PIE: [[DATA]] {{.*}} R_AARCH64_RELATIVE
# IPLT: [[GOTPLT]] {{.*}} R_AARCH64_IRELATIVE
# IPLT-PDE: Hex dump of section '.data':
# IPLT-PDE-NEXT: 80012100

#--- a.s
.globl ifunc
.type ifunc, @gnu_indirect_function
ifunc:
ret

.data
.quad ifunc

#--- b.s
adrp x0, ifunc
add x0, x0, :lo12:ifunc
