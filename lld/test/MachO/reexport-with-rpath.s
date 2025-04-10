# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir -p %t/cc/two/three
# RUN: mkdir -p %t/bb/two/three
# RUN: mkdir -p %t/aa/two/three

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/c.s -o %t/c.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/a.s -o %t/a.o

# RUN: %lld -dylib -install_name @rpath/two/three/libCee.dylib %t/c.o -o %t/cc/two/three/libCee.dylib 
# RUN: %lld -dylib -install_name @rpath/two/three/libBee.dylib -L%t/cc/two/three -sub_library libCee -lCee %t/b.o -o %t/bb/two/three/libBee.dylib -rpath %t/cc
# RUN: %lld -dylib -install_name @rpath/two/three/libAee.dylib -L%t/bb/two/three -sub_library libBee -lBee %t/a.o -o %t/aa/two/three/libAee.dylib

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
