	.section	".text"
	.comm	storeVal_vector,8,8
	.comm	useVal_vector,8,8
	.globl storeVal_longlong, useAddr_longlong, useVal_longlong, storeVal_sshort
	.globl useAddr_sshort, useVal_sshort, storeVal_sint, useAddr_sint, useVal_sint
	.globl storeVal_double, useAddr_double, useVal_double, storeVal_float
	.globl useAddr_float, useVal_float, storeVal_uint, storeVal_uint
	.globl useVal_uint, storeVal_ushort, useAddr_ushort, useVal_ushort
	.globl storeVal, useAddr, useVal
	.section	".data"
	.align 3
	.type	storeVal_longlong, @object
	.size	storeVal_longlong, 8
storeVal_longlong:
	.quad	18
useAddr_longlong:
	.quad	17
useVal_longlong:
	.quad	16
storeVal_sshort:
	.short	-15
useAddr_sshort:
	.short	-14
useVal_sshort:
	.short	-13
	.zero	2
storeVal_sint:
	.long	-12
useAddr_sint:
	.long	-11
useVal_sint:
	.long	-10
	.zero	4
storeVal_double:
	.long	858993459
	.long	1076966195
useAddr_double:
	.long	-1717986918
	.long	-1070589543
useVal_double:
	.long	0
	.long	1076756480
storeVal_float:
	.long	1045220557
useAddr_float:
	.long	-1050568294
useVal_float:
	.long	1095761920
storeVal_uint:
	.long	12
useAddr_uint:
	.long	11
useVal_uint:
	.long	10
storeVal_ushort:
	.short	1
useAddr_ushort:
	.short	10
useVal_ushort:
	.short	5
storeVal:
	.byte	-1
useAddr:
	.byte	10
useVal:
	.byte	5
