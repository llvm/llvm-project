# REQUIRES: x86

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/basic.s -o %t.basic.o
# RUN: %lld %t.basic.o -o %t.basic
# RUN: llvm-objdump --syms --macho --function-starts %t.basic | FileCheck %s --check-prefix=BASIC

# BASIC:      [[#%,MAIN:]]
# BASIC-NEXT: [[#%,F1:]]
# BASIC-NEXT: [[#%,F2:]]
# BASIC-NEXT: SYMBOL TABLE:
# BASIC: [[#MAIN]] {{.*}} _main
# BASIC: [[#F1]] {{.*}} _f1
# BASIC: [[#F2]] {{.*}} _f2

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/alias.s -o %t.alias.o
# RUN: %lld %t.alias.o -o %t.alias
# RUN: llvm-objdump --syms --macho --function-starts %t.alias | FileCheck %s --check-prefix=ALIAS

# ALIAS:      [[#%,MAIN:]]
# ALIAS-NEXT: [[#%,F1:]]
# ALIAS-NEXT: SYMBOL TABLE:
# ALIAS: [[#MAIN]] {{.*}} _main
# ALIAS: [[#F1]] {{.*}} _f1

#--- basic.s
.section  __TEXT,__text,regular,pure_instructions
.globl  _f1
.globl  _f2
.globl  _main
_f1:
  retq
_f2:
  retq
_main:
  retq

#--- alias.s
.section  __TEXT,__text,regular,pure_instructions
.globl  _f1
.equiv  _f2, _f1
.globl  _main
_f1:
  retq
_main:
  retq
