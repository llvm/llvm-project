# REQUIRES: x86
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libtlv.s -o %t/libtlv.o
# RUN: %lld -dylib -install_name @executable_path/libtlv.dylib \
# RUN:   -lSystem -o %t/libtlv.dylib %t/libtlv.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: not %lld -lSystem -L%t -ltlv -o /dev/null %t/test.o 2>&1 | FileCheck %s -DFILE=%t/test.o

# CHECK: error: GOT_LOAD relocation requires that variable not be thread-local for `_foo' in [[FILE]]:(__text)

#--- libtlv.s
.section	__DATA,__thread_vars,thread_local_variables
.globl _foo
_foo:

#--- test.s
.text
.globl _main
_main:
  movq _foo@GOTPCREL(%rip), %rax
  ret
