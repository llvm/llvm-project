# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/calls-foo.s -o %t/calls-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/1.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/common.s -o %t/common.o

# RUN: echo "%t/1.o" > %t/filelist
# RUN: echo "%t/2.o" >> %t/filelist

# RUN: llvm-as %t/1.ll -o %t/1.bc
# RUN: llvm-as %t/2.ll -o %t/2.bc

## Neither 1.o nor 2.o is loaded.
# RUN: %lld %t/main.o -o %t/out --start-lib %t/1.o %t/2.o --end-lib -why_load | count 0
# RUN: %lld %t/main.o -o %t/out --start-lib -filelist %t/filelist --end-lib -why_load | count 0
# RUN: llvm-readobj -s %t/out | FileCheck %s
# CHECK-NOT: Name: _foo
# CHECK-NOT: Name: _bar

## _bar loads 2.o. The last --end-lib can be omitted.
# RUN: %lld %t/main.o -o %t/out -u _bar --start-lib %t/1.o %t/2.o -t -why_load | FileCheck %s --check-prefix=CHECK2WHY
# RUN: %lld %t/main.o -o %t/out -u _bar --start-lib -filelist %t/filelist -t -why_load | FileCheck %s --check-prefix=CHECK2WHY
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=CHECK2 %s
# CHECK2WHY:      {{.*}}main.o
# CHECK2WHY-NEXT: {{.*}}2.o
# CHECK2WHY-NEXT: _bar forced load of {{.*}}2.o
# CHECK2WHY-EMPTY:
# CHECK2-NOT: Name: _foo
# CHECK2: Name: _bar
# CHECK2-NOT: Name: _foo

## _foo loads 1.o. 1.o loads 2.o.
# RUN: %lld %t/main.o -o %t/out -u _foo --start-lib %t/1.o %t/2.o -why_load | FileCheck %s --check-prefix=CHECK3WHY
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=CHECK3 %s
# RUN: %lld %t/main.o -o %t/out -u _foo --start-lib %t/2.o --end-lib --start-lib %t/1.o -why_load | FileCheck %s --check-prefix=CHECK3WHY
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=CHECK3 %s
# CHECK3WHY:      _foo forced load of {{.*}}1.o
# CHECK3WHY-NEXT: _bar forced load of {{.*}}2.o
# CHECK3WHY-EMPTY:
# CHECK3-DAG: Name: _foo
# CHECK3-DAG: Name: _bar

## Don't treat undefined _bar in 1.o as a lazy definition.
# RUN: not %lld %t/main.o -u _bar --start-lib %t/1.o 2>&1 | FileCheck %s --check-prefix=CHECK4
# CHECK4: error: undefined symbol: _bar

# RUN: %lld %t/main.o -o %t/out -u _common --start-lib %t/common.o
# RUN: llvm-readobj -s %t/out | FileCheck %s --check-prefix=COMMON1
# COMMON1: Name: _common

# RUN: %lld %t/main.o -o %t/out --start-lib %t/common.o
# RUN: llvm-readobj -s %t/out | FileCheck %s --check-prefix=COMMON2
# COMMON2-NOT: Name: _common

## Neither 1.bc nor 2.bc is loaded.
# RUN: %lld %t/main.o -o %t/out --start-lib %t/1.bc %t/2.bc -why_load | count 0
# RUN: llvm-readobj -s %t/out | FileCheck %s --check-prefix=BITCODE
# BITCODE-NOT: Name: _foo
# BITCODE-NOT: Name: _bar

## _bar loads 2.bc.
# RUN: %lld %t/main.o -o %t/out -u _bar --start-lib %t/1.bc %t/2.bc -why_load | FileCheck %s --check-prefix=BITCODE2WHY
# RUN: llvm-readobj -s %t/out | FileCheck %s --check-prefix=BITCODE2
# BITCODE2WHY:      _bar forced load of {{.*}}2.bc
# BITCODE2WHY-EMPTY:
# BITCODE2-NOT: Name: _foo
# BITCODE2: Name: _bar
# BITCODE2-NOT: Name: _foo

## calls-foo.o loads 1.bc. 1.bc loads 2.bc.
# RUN: %lld %t/calls-foo.o -o %t/out --start-lib %t/1.bc %t/2.bc -why_load | FileCheck %s --check-prefix=BITCODE3WHY
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=BITCODE3 %s
# RUN: %lld %t/calls-foo.o -o %t/out --start-lib %t/2.bc --end-lib --start-lib %t/1.bc -why_load | FileCheck %s --check-prefix=BITCODE3WHY
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=BITCODE3 %s
# BITCODE3WHY:      _foo forced load of {{.*}}1.bc
# BITCODE3WHY-NEXT: _bar forced load of {{.*}}2.bc
# BITCODE3WHY-EMPTY:
# BITCODE3-DAG: Name: _foo

# RUN: not %lld %t/main.o --start-lib --start-lib 2>&1 | FileCheck -check-prefix=NESTED-LIB %s
# NESTED-LIB: error: nested --start-lib

# RUN: not %lld --end-lib 2>&1 | FileCheck %s --check-prefix=STRAY
# STRAY: error: stray --end-lib

# RUN: %lld -dylib --start-lib %t/1.bc %t/2.o --end-lib -all_load -o %t/out
# RUN: llvm-readobj -s %t/out | FileCheck --check-prefix=ALL-LOAD %s
# ALL-LOAD-DAG: _foo
# ALL-LOAD-DAG: _bar

#--- main.s
.globl _main
_main:

#--- calls-foo.s
.globl _main
_main:
  call _foo

#--- 1.s
.globl _foo
_foo:
  call _bar

#--- 2.s
.globl _bar
_bar:
  ret

#--- common.s
.comm _common, 1

#--- 1.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  tail call void () @bar()
  ret void
}

declare void @bar()

#--- 2.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @bar() {
  ret void
}
