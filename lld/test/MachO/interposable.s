# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/3.s -o %t/3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -arch x86_64 -interposable -lSystem -o %t/main %t/main.o %t/2.o %t/3.o
# RUN: llvm-objdump --macho -d %t/main | FileCheck %s --check-prefix BUNDLE-OBJ
BUNDLE-OBJ-LABEL: _my_user:
BUNDLE-OBJ-NEXT:          callq   [[#%#x,]] ## symbol stub for: _my_friend

#--- 2.s
# my_lib: This contains the exported function
.globl _my_friend
_my_friend:
  retq

#--- 3.s
# _my_user.s: This is the user/caller of the
#             exported function
.text
_my_user:
  callq _my_friend()
  retq

#--- main.s
# main.s: dummy exec/main loads the exported function.
# This is basically a way to say `my_user` should get
# `my_func` from this executable.
.globl _main
.text
 _main:
  retq
