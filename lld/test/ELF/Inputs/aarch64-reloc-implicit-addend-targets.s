.globl abs16
.globl abs32
.globl abs64
.globl big64
.globl pcrel
.globl data
.globl branchtarget
.globl calltarget

.equ abs16, 0x9999
.equ data, 0x100000
.equ branchtarget, 0x200100
.equ calltarget, 0x02000100
.equ pcrel, 0x245678
.equ abs32, 0x88888888
.equ abs64, 0x7777777777777777
.equ big64, 0x77ffffffffffff77
