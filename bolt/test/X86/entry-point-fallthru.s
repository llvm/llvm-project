## Checks that fallthroughs spanning entry points are accepted in aggregation
## mode.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: link_fdata %s %t %t.preagg PREAGG
# RUN: perf2bolt %t -p %t.preagg --pa -o %t.fdata | FileCheck %s
# CHECK: traces mismatching disassembled function contents: 0
# RUN: FileCheck %s --check-prefix=CHECK-FDATA --input-file %t.fdata
# CHECK-FDATA:      1 main 0 1 main 6 0 1
# CHECK-FDATA-NEXT: 1 main e 1 main 11 0 1
# CHECK-FDATA-NEXT: 1 main 11 1 main 0 0 1

	.globl main
main:
	.cfi_startproc
	vmovaps %zmm31,%zmm3

next:
	add    $0x4,%r9
	add    $0x40,%r10
	dec    %r14
Ljmp:
	jne    main
# PREAGG: T #Ljmp# #main# #Ljmp# 1
	ret
	.cfi_endproc
.size main,.-main
