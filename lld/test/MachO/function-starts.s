# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/basic.s -o %t/basic.o
# RUN: %lld -lSystem %t/basic.o -o %t/basic
# RUN: llvm-objdump --syms %t/basic > %t/objdump
# RUN: llvm-objdump --macho --function-starts %t/basic >> %t/objdump
# RUN: FileCheck %s --check-prefix=BASIC < %t/objdump

# BASIC-LABEL: SYMBOL TABLE:
# BASIC-NEXT:  [[#%x,MAIN:]] g F __TEXT,__text _main
# BASIC-NEXT:  [[#%x,F1:]] g F __TEXT,__text _f1
# BASIC-NEXT:  [[#%x,F2:]] g F __TEXT,__text _f2
# BASIC-LABEL: basic:
# BASIC:       [[#F1]]
# BASIC:       [[#F2]]
# BASIC:       [[#MAIN]]

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/alias.s -o %t/alias.o
# RUN: %lld -lSystem %t/alias.o -o %t/alias
# RUN: llvm-objdump --syms  %t/alias > %t/objdump
# RUN: llvm-objdump --macho --function-starts %t/alias >> %t/objdump
# RUN: FileCheck %s --check-prefix=ALIAS < %t/objdump

# ALIAS-LABEL: SYMBOL TABLE:
# ALIAS-NEXT:  [[#%x,F2:]] l F __TEXT,__text _f2
# ALIAS-NEXT:  [[#%x,MAIN:]] g F __TEXT,__text _main
# ALIAS-NEXT:  [[#%x,F1:]] g F __TEXT,__text _f1
# ALIAS-LABEL: alias:
# ALIAS:       [[#F1]]
# ALIAS:       [[#MAIN]]

# RUN: %lld %t/basic.o -no_function_starts -o %t/basic-no-function-starts
# RUN: llvm-objdump --macho --function-starts %t/basic-no-function-starts | FileCheck %s --check-prefix=NO-FUNCTION-STARTS
# NO-FUNCTION-STARTS: basic-no-function-starts:
# NO-FUNCTION-STARTS-EMPTY:

# RUN: %lld -lSystem %t/basic.o -no_function_starts -function_starts -o %t/basic-explicit
# RUN: llvm-objdump --syms %t/basic > %t/objdump
# RUN: llvm-objdump --macho --function-starts %t/basic >> %t/objdump
# RUN: FileCheck %s --check-prefix=BASIC < %t/objdump

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/local.s -o %t/local.o
# RUN: %lld -lSystem %t/local.o -o %t/local
# RUN: llvm-objdump --syms %t/local > %t/objdump
# RUN: llvm-objdump --macho --function-starts %t/local >> %t/objdump
# RUN: FileCheck %s --check-prefix=LOCAL < %t/objdump

# LOCAL-LABEL: SYMBOL TABLE:
# LOCAL-NEXT:  [[#%x,F1:]] l F __TEXT,__text +[Foo bar]
# LOCAL-NEXT:  [[#%x,F2:]] l F __TEXT,__text _foo
# LOCAL:       [[#%x,F3:]] g F __TEXT,__text _main
# LOCAL-LABEL: local:
# LOCAL:       [[#F1]]
# LOCAL:       [[#F2]]
# LOCAL:       [[#F3]]

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

#--- local.s
.section __TEXT,__text
"+[Foo bar]":
  retq

_foo:
  retq

.globl _main
_main:
  retq
