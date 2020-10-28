# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

## This test checks the LLD implementation of the Local Exec TLS model
## when using prefixed instructions like paddi.

# SYMBOL:      Symbol table '.symtab' contains 6 entries:
# SYMBOL:      3: 0000000000000000     0 TLS     LOCAL DEFAULT     2 x
# SYMBOL-NEXT: 4: 0000000000000004     0 TLS     LOCAL DEFAULT     2 y
# SYMBOL-NEXT: 5: 0000000000000008     0 TLS     LOCAL DEFAULT     2 z

# CHECK-LABEL: <LocalExecAddr>:
# CHECK:       paddi 3, 13, -28672, 0
# CHECK-NEXT:  paddi 3, 13, -28668, 0
# CHECK-NEXT:  paddi 3, 13, -28652, 0
# CHECK-NEXT:  blr

# CHECK-LABEL: <LocalExecVal>:
# CHECK:       paddi 3, 13, -28672, 0
# CHECK-NEXT:  lwz 3, 0(3)
# CHECK-NEXT:  paddi 3, 13, -28668, 0
# CHECK-NEXT:  lwz 3, 0(3)
# CHECK-NEXT:  paddi 3, 13, -28652, 0
# CHECK-NEXT:  lwz 3, 0(3)
# CHECK-NEXT:  blr

LocalExecAddr:
	paddi 3, 13, x@TPREL, 0
	paddi 3, 13, y@TPREL, 0
	paddi 3, 13, z@TPREL+12, 0
	blr

LocalExecVal:
	paddi 3, 13, x@TPREL, 0
	lwz 3, 0(3)
	paddi 3, 13, y@TPREL, 0
	lwz 3, 0(3)
	paddi 3, 13, z@TPREL+12, 0
	lwz 3, 0(3)
	blr

.section .tbss, "awT", @nobits
x:
	.long	0
y:
	.long	0
z:
	.space	20
