# REQUIRES: x86
## This tests that lld handles (re)exported libraries with rpath properly .
## The tests are as follows:
## - libAee re-exports libBee while defining the install-path with @rpath variable
##   - libBee re-exports libCee while defining the install-path with @rpath variable
##     - libCee contains _c_func symbol.
##
## We check that when linking a main which references _c_func, lld can find the symbol.

# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir -p %t/cc/two/three
# RUN: mkdir -p %t/bb/two/three
# RUN: mkdir -p %t/aa/two/three

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c.s -o %t/c.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -dylib -install_name @rpath/two/three/libCee.dylib %t/c.o -o %t/cc/two/three/libCee.dylib 
# RUN: %lld -dylib -install_name @rpath/two/three/libBee.dylib -L%t/cc/two/three -sub_library libCee -lCee %t/b.o -o %t/bb/two/three/libBee.dylib -rpath %t/cc
# RUN: %lld -dylib -install_name @rpath/two/three/libAee.dylib -L%t/bb/two/three -sub_library libBee -lBee %t/a.o -o %t/aa/two/three/libAee.dylib -rpath %t/aa
# RUN: %lld %t/main.o  -L%t/aa/two/three -lAee -o %t/a.out -rpath %t/aa -rpath %t/bb -rpath %t/cc

#--- c.s
.text
.global _c_func
_c_func:
  mov $0, %rax
  ret

#--- b.s
.text
.global _b_func

_b_func:
  mov $0, %rax
  ret

#--- a.s
.text
.global _a_func
_a_func:
  mov $0, %rax
  ret

#--- main.s
.globl _main
_main:
  callq _c_func
  callq _b_func
  callq _a_func
  ret
