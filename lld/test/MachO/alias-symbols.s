# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/aliases.s -o %t/aliases.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/definitions.s -o %t/definitions.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-extern-alias-to-weak.s -o %t/weak-extern-alias-to-weak.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-extern-alias-to-strong.s -o %t/weak-extern-alias-to-strong.o

# RUN: %lld -lSystem %t/aliases.o %t/definitions.o -o %t/out
# RUN: llvm-objdump --macho --syms %t/out | FileCheck %s

## local aliases should be dropped entirely. --implicit-check-not doesn't seem
## to work well with -DAG matches, so we check for _local_alias' absence in a
## separate step.
# RUN: llvm-objdump --macho --syms %t/out | FileCheck /dev/null --implicit-check-not _local_alias

# CHECK-DAG: [[#%.16x,STRONG:]] g     F __TEXT,__text _strong
# CHECK-DAG: [[#%.16x,WEAK_1:]]  w    F __TEXT,__text _weak_1
# CHECK-DAG: [[#%.16x,PEXT:]]   l     F __TEXT,__text _pext
# CHECK-DAG: [[#%.16x,DEAD:]]   g     F __TEXT,__text _dead
# CHECK-DAG: [[#STRONG]]        l     F __TEXT,__text _pext_alias
# CHECK-DAG: [[#PEXT]]          l     F __TEXT,__text _alias_to_pext
# CHECK-DAG: [[#STRONG]]        g     F __TEXT,__text _extern_alias_to_strong
# CHECK-DAG: [[#WEAK_1]]         w    F __TEXT,__text _weak_extern_alias_to_weak
# CHECK-DAG: [[#DEAD]]          g     F __TEXT,__text _no_dead_strip_alias
# CHECK-DAG: [[#STRONG]]        g     F __TEXT,__text _weak_extern_alias_to_strong

# RUN: %lld -lSystem -dead_strip %t/aliases.o %t/definitions.o -o %t/dead-stripped
# RUN: llvm-objdump --macho --syms %t/dead-stripped | FileCheck %s --check-prefix=STRIPPED

# STRIPPED:       SYMBOL TABLE:
# STRIPPED-NEXT:  g     F __TEXT,__text _main
# STRIPPED-NEXT:  g     F __TEXT,__text __mh_execute_header
# STRIPPED-NEXT:          *UND* dyld_stub_binder
# STRIPPED-EMPTY:

# RUN: not %lld -lSystem %t/aliases.o %t/definitions.o \
# RUN:   %t/weak-extern-alias-to-strong.o -o /dev/null 2>&1

## Verify that we preserve the file names of the aliases, rather than using the
## filename of the aliased symbols.
# DUP:      error: duplicate symbol: _weak_extern_alias_to_weak
# DUP-NEXT: >>> defined in {{.*}}aliases.o
# DUP-NEXT: >>> defined in {{.*}}weak-extern-alias-to-weak.o

## The following cases are actually all dup symbol errors under ld64. Alias
## symbols are treated like strong extern symbols by ld64 even if the symbol they alias
## is actually weak. LLD OTOH does not check for dup symbols until after
## resolving the aliases; this makes for a simpler implementation.
## The following test cases are meant to elucidate what LLD's behavior is, but
## we should feel free to change it in the future should it be helpful for the
## implementation.

# RUN: %lld -lSystem %t/aliases.o %t/definitions.o \
# RUN:   %t/weak-extern-alias-to-weak.o -o %t/alias-clash-1
# RUN: llvm-objdump --macho --syms %t/alias-clash-1 | FileCheck %s --check-prefix WEAK-1

# RUN: %lld -lSystem %t/weak-extern-alias-to-weak.o %t/aliases.o \
# RUN:   %t/definitions.o -o %t/alias-clash-2
# RUN: llvm-objdump --macho --syms %t/alias-clash-2 | FileCheck %s --check-prefix WEAK-2

# RUN: %lld -lSystem %t/aliases.o %t/definitions.o \
# RUN:   -alias _weak_2 _weak_extern_alias_to_weak -o %t/opt-vs-symbol
# RUN: llvm-objdump --macho --syms %t/opt-vs-symbol | FileCheck %s --check-prefix WEAK-2

# RUN: %lld -lSystem -alias _weak_2 _weak_extern_alias_to_weak %t/aliases.o \
# RUN:   %t/definitions.o -o %t/opt-vs-symbol
# RUN: llvm-objdump --macho --syms %t/opt-vs-symbol | FileCheck %s --check-prefix WEAK-2

# WEAK-1-DAG: [[#%.16x,WEAK_1:]]  w    F __TEXT,__text _weak_1
# WEAK-1-DAG: [[#WEAK_1]]         w    F __TEXT,__text _weak_extern_alias_to_weak

# WEAK-2-DAG: [[#%.16x,WEAK_2:]]  w    F __TEXT,__text _weak_2
# WEAK-2-DAG: [[#WEAK_2]]         w    F __TEXT,__text _weak_extern_alias_to_weak

#--- aliases.s
.globl _extern_alias_to_strong, _weak_extern_alias_to_weak
.weak_definition _weak_extern_alias_to_weak

## Private extern aliases result in local symbols in the output (i.e. it is as
## if the aliased symbol is also private extern.)
.private_extern _pext_alias

## This test case demonstrates that it doesn't matter whether the alias itself
## is strong or weak. Rather, what matters is whether the aliased symbol is
## strong or weak.
.globl _weak_extern_alias_to_strong
.weak_definition _weak_extern_alias_to_strong

## no_dead_strip doesn't retain the aliased symbol if it is dead
.globl _no_dead_strip_alias
.no_dead_strip _no_dead_strip_alias

.globl _alias_to_pext
_alias_to_pext = _pext

_extern_alias_to_strong = _strong
_weak_extern_alias_to_weak = _weak_1
_weak_extern_alias_to_strong = _strong

_pext_alias = _strong
_local_alias = _strong
_no_dead_strip_alias = _dead

.subsections_via_symbols

#--- weak-extern-alias-to-weak.s
.globl _weak_extern_alias_to_weak
.weak_definition _weak_extern_alias_to_weak
_weak_extern_alias_to_weak = _weak_2

#--- weak-extern-alias-to-strong.s
.globl _weak_extern_alias_to_strong
.weak_definition _weak_extern_alias_to_strong
_weak_extern_alias_to_strong = _strong

#--- definitions.s
.globl _strong, _weak_1, _weak_2, _dead
.private_extern _pext
.weak_definition _weak_1
.weak_definition _weak_2

_strong:
  .space 1
_weak_1:
  .space 1
_weak_2:
  .space 1
_dead:
  .space 1
_pext:
  .space 1

.globl _main
_main:

.subsections_via_symbols
