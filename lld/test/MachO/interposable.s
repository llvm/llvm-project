
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/3.s -o %t/3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -lSystem -o %t/main %t/main.o %t/2.o %t/3.o
# RUN: llvm-objdump  --macho --lazy-bind %t/main | FileCheck %s --check-prefix BASE-OBJ
# BASE-OBJ: segment  section             address            dylib                 symbol
# BASE-OBJ-EMPTY:

# RUN: %lld -interposable -lSystem -o %t/main %t/main.o %t/2.o %t/3.o
# RUN: llvm-objdump  --macho --lazy-bind %t/main | FileCheck %s --check-prefix INTER-OBJ
# INTER-OBJ: segment  section             address            dylib                 symbol
# INTER-OBJ-NEXT: __DATA   __la_symbol_ptr     0x[[#%x,]]    flat-namespace        _main
# INTER-OBJ: __DATA   __la_symbol_ptr     0x[[#%x,]]    flat-namespace        my_func
# INTER-OBJ-EMPTY:

#--- 2.s
# my_lib: This contains the exported function
.globl my_func
my_func:
  retq

#--- 3.s
# my_user.s: This is the user/caller of the
#            exported function
.text
my_user:
  callq my_func()
  retq

#--- main.s
# main.s: dummy exec/main loads the exported function.
# This is basically a way to say `my_user` should get
# `my_func` from this executable.
.globl _main
.text
 _main:
  retq
