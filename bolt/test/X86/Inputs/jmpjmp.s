	.text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 0
	.cfi_startproc
.LBB06:
	callq	testfunc
	retq
	.cfi_endproc
.size main, .-main

  .globl testfunc
  .type testfunc, %function
testfunc:
# FDATA: 0 [unknown] 0 1 testfunc 0 0 0
	.cfi_startproc
.LBB07:
.LBB07_br: 	jmp	.Ltmp6
# FDATA: 1 testfunc #.LBB07_br# 1 testfunc #.Ltmp6# 0 0

.LFT1:
.LFT1_br: 	jmp	.Ltmp7
# FDATA: 1 testfunc #.LFT1_br# 1 testfunc #.Ltmp7# 0 0

.Ltmp6:
.Ltmp6_br: 	jmp	.Ltmp8
# FDATA: 1 testfunc #.Ltmp6_br# 1 testfunc #.Ltmp8# 0 0

.Ltmp7:
# FDATA: 1 testfunc #.Ltmp7_br# 1 testfunc #.Ltmp8# 0 0

.Ltmp8:
	retq
	.cfi_endproc
.size testfunc, .-testfunc
