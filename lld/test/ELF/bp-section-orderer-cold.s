# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-profdata merge a.proftext -o a.profdata

## Compression sort: verify isColdSection identifies .text.unlikely and
## .text.split sections by checking the verbose cold function count.
# RUN: ld.lld a.o -o compr.out --bp-compression-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=COMPRESSION

# COMPRESSION: Functions for compression: 4
# COMPRESSION: Cold functions for compression: 3

## Startup sort + compression sort: cold1 is in the startup trace so it
## doesn't count toward either compression bucket.
# RUN: ld.lld a.o -o startup-compr.out --irpgo-profile=a.profdata --bp-startup-sort=function --bp-compression-sort=function --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=STARTUP-COMPR

# STARTUP-COMPR: Functions for startup: 2
# STARTUP-COMPR: Functions for compression: 3
# STARTUP-COMPR: Cold functions for compression: 2

#--- a.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size:
1
# Weight
1
hot1, cold1

hot1
# Func Hash:
1111
# Num Counters:
1
# Counter Values:
1

cold1
# Func Hash:
2222
# Num Counters:
1
# Counter Values:
1

#--- a.s
	.section	.text.main,"ax",@progbits
	.globl	main
	.type	main,@function
main:
	mov	w0, wzr
	ret
.Lfunc_end_main:
	.size	main, .Lfunc_end_main-main

	.section	.text.unlikely.cold1,"ax",@progbits
	.globl	cold1
	.type	cold1,@function
cold1:
	add	w0, w0, #10
	add	w1, w1, #11
	bl	main
	ret
.Lfunc_end_cold1:
	.size	cold1, .Lfunc_end_cold1-cold1

	.section	.text.hot1,"ax",@progbits
	.globl	hot1
	.type	hot1,@function
hot1:
	add	w0, w0, #1
	add	w1, w1, #2
	bl	main
	ret
.Lfunc_end_hot1:
	.size	hot1, .Lfunc_end_hot1-hot1

	.section	.text.unlikely.cold2,"ax",@progbits
	.globl	cold2
	.type	cold2,@function
cold2:
	add	w0, w0, #20
	add	w1, w1, #21
	bl	hot1
	ret
.Lfunc_end_cold2:
	.size	cold2, .Lfunc_end_cold2-cold2

	.section	.text.hot2,"ax",@progbits
	.globl	hot2
	.type	hot2,@function
hot2:
	add	w0, w0, #2
	add	w1, w1, #3
	bl	hot1
	ret
.Lfunc_end_hot2:
	.size	hot2, .Lfunc_end_hot2-hot2

	.section	.text.split.cold_split1,"ax",@progbits
	.globl	cold_split1
	.type	cold_split1,@function
cold_split1:
	add	w0, w0, #30
	add	w1, w1, #31
	bl	cold1
	ret
.Lfunc_end_cold_split1:
	.size	cold_split1, .Lfunc_end_cold_split1-cold_split1

	.section	.text.hot3,"ax",@progbits
	.globl	hot3
	.type	hot3,@function
hot3:
	add	w0, w0, #3
	add	w1, w1, #4
	bl	cold1
	ret
.Lfunc_end_hot3:
	.size	hot3, .Lfunc_end_hot3-hot3

	.section	".note.GNU-stack","",@progbits
