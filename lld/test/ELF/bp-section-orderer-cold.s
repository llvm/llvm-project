# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: llvm-profdata merge a.proftext -o a.profdata

## Simple glob: all .text* sections in one group.
# RUN: ld.lld a.o -o simple.out \
# RUN:   --bp-compression-sort-section=".text*" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=SIMPLE

## Deprecated --bp-compression-sort=function still works.
# RUN: ld.lld a.o -o deprecated.out \
# RUN:   --bp-compression-sort=function \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LEGACY-FUNCTION

# SIMPLE: Sections for compression: 7
# SIMPLE: Compression groups: 1
# SIMPLE:   .text*: 7 sections

# LEGACY-FUNCTION: Sections for compression: 7
# LEGACY-FUNCTION: Compression groups: 1
# LEGACY-FUNCTION:   legacy:function: 7 sections

## Layout priority: .text.unlikely*, .text*, .text.split*
# RUN: ld.lld a.o -o layout.out \
# RUN:   --bp-compression-sort-section=".text*=1" \
# RUN:   --bp-compression-sort-section=".text.unlikely*=0" \
# RUN:   --bp-compression-sort-section=".text.split*=2" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LAYOUT
# RUN: llvm-nm -jn layout.out | FileCheck %s --check-prefix=LAYOUT-ORDER

# LAYOUT: Sections for compression: 7
# LAYOUT: Compression groups: 3
# LAYOUT:   .text.unlikely*: 2 sections
# LAYOUT:   .text*: 4 sections
# LAYOUT:   .text.split*: 1 sections

# LAYOUT-ORDER: cold1
# LAYOUT-ORDER: hot1
# LAYOUT-ORDER: cold_split1

## Match priority: explicit match_priority wins
# RUN: ld.lld a.o -o match.out \
# RUN:   --bp-compression-sort-section=".text.unlikely*=2=0" \
# RUN:   --bp-compression-sort-section=".text*=0" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=MATCH
# RUN: llvm-nm -jn match.out | FileCheck %s --check-prefix=MATCH-ORDER

## Match priority: lower match_priority wins
# RUN: ld.lld a.o -o match-prio.out \
# RUN:   --bp-compression-sort-section=".text*=0=1" \
# RUN:   --bp-compression-sort-section=".text.unlikely*=2=0" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=MATCH
# RUN: llvm-nm -jn match-prio.out | FileCheck %s --check-prefix=MATCH-ORDER

# MATCH: Sections for compression: 7
# MATCH: Compression groups: 2
# MATCH:   .text*: 5 sections
# MATCH:   .text.unlikely*: 2 sections
# MATCH-ORDER: main
# MATCH-ORDER: cold1

## Match priority tie: last match wins
# RUN: ld.lld a.o -o match-tie.out \
# RUN:   --bp-compression-sort-section=".text.unlikely*=0=0" \
# RUN:   --bp-compression-sort-section=".text*=0=0" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=MATCH-TIE

# MATCH-TIE: Compression groups: 1
# MATCH-TIE:   .text*: 7 sections

## Layout priority tie: groups ordered by glob string
# RUN: ld.lld a.o -o layout-tie.out \
# RUN:   --bp-compression-sort-section=".text.unlikely*=0=1" \
# RUN:   --bp-compression-sort-section=".text.split*=0=2" \
# RUN:   --bp-compression-sort-section=".text*=0" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=LAYOUT-TIE
# RUN: llvm-nm -jn layout-tie.out | FileCheck %s --check-prefix=LAYOUT-TIE-ORDER

# LAYOUT-TIE: Compression groups: 3
# LAYOUT-TIE:   .text*: 4 sections
# LAYOUT-TIE:   .text.split*: 1 sections
# LAYOUT-TIE:   .text.unlikely*: 2 sections
# LAYOUT-TIE-ORDER: main
# LAYOUT-TIE-ORDER: cold_split1
# LAYOUT-TIE-ORDER: cold1

## Startup sort + compression: startup, .text*, .text.unlikely*, .text.split*
# RUN: ld.lld a.o -o startup-compr.out --irpgo-profile=a.profdata \
# RUN:   --bp-startup-sort=function \
# RUN:   --bp-compression-sort-section=".text*=0" \
# RUN:   --bp-compression-sort-section=".text.unlikely*=1=1" \
# RUN:   --bp-compression-sort-section=".text.split*=2=1" \
# RUN:   --verbose-bp-section-orderer 2>&1 | FileCheck %s --check-prefix=STARTUP
# RUN: llvm-nm -jn startup-compr.out | FileCheck %s --check-prefix=STARTUP-ORDER

# STARTUP: Functions for startup: 2
# STARTUP: Sections for compression: 5
# STARTUP: Compression groups: 3
# STARTUP-ORDER: hot1
# STARTUP-ORDER: main
# STARTUP-ORDER: cold2
# STARTUP-ORDER: cold_split1

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
	.section	.text.split.cold_split1,"ax",@progbits
	.globl	cold_split1
	.type	cold_split1,@function
cold_split1:
	add	w0, w0, #30
	add	w1, w1, #31
	bl	main
	ret
.Lfunc_end_cold_split1:
	.size	cold_split1, .Lfunc_end_cold_split1-cold_split1

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

	.section	.text.main,"ax",@progbits
	.globl	main
	.type	main,@function
main:
	mov	w0, wzr
	ret
.Lfunc_end_main:
	.size	main, .Lfunc_end_main-main

	.section	".note.GNU-stack","",@progbits
