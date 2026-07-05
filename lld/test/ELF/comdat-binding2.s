# REQUIRES: x86
## Test we don't report duplicate definition errors when mixing Clang STB_WEAK
## and GCC STB_GNU_UNIQUE symbols.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 weak.s -o weak.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 unique.s -o unique.o
# RUN: ld.lld weak.o unique.o -o weak
# RUN: llvm-readelf -s weak | FileCheck %s --check-prefix=WEAK
# RUN: ld.lld unique.o weak.o -o unique
# RUN: llvm-readelf -s unique | FileCheck %s --check-prefix=UNIQUE

# WEAK:   OBJECT  WEAK   DEFAULT [[#]] _ZN1BIiE1aE
# UNIQUE: OBJECT  UNIQUE DEFAULT [[#]] _ZN1BIiE1aE

#--- weak.s
## Clang
	.type	_ZN1BIiE1aE,@object
	.section	.bss._ZN1BIiE1aE,"aGwR",@nobits,_ZN1BIiE1aE,comdat
	.weak	_ZN1BIiE1aE
_ZN1BIiE1aE:
	.zero	4

	.type	_ZGVN1BIiE1aE,@object
	.section	.bss._ZGVN1BIiE1aE,"aGw",@nobits,_ZN1BIiE1aE,comdat
	.weak	_ZGVN1BIiE1aE
_ZGVN1BIiE1aE:
	.quad	0

#--- unique.s
## GCC -fgnu-unique. Note the different group signature for the second group.
	.weak	_ZN1BIiE1aE
	.section	.bss._ZN1BIiE1aE,"awG",@nobits,_ZN1BIiE1aE,comdat
	.type	_ZN1BIiE1aE, @gnu_unique_object
_ZN1BIiE1aE:
	.zero	4

	.weak	_ZGVN1BIiE1aE
	.section	.bss._ZGVN1BIiE1aE,"awG",@nobits,_ZGVN1BIiE1aE,comdat
	.type	_ZGVN1BIiE1aE, @gnu_unique_object
_ZGVN1BIiE1aE:
	.zero	8
