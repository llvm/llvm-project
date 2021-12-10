# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: split-file %s %t.dir
# RUN: cd %t.dir

## Verify that in the following case:
##
##   <exec>
##   +- ref.so (weak reference to foo)
##   +- wrap.so (non-weak reference to foo)
##      +- def.so (defines foo)
##
## we don't report that foo is undefined in ref.so when linking <exec>.

# RUN: llvm-mc -filetype=obj -triple=x86_64 ref.s -o ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 wrap.s -o wrap.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 def.s -o def.o
# RUN: ld.lld -shared ref.o -o ref.so
# RUN: ld.lld -shared def.o -soname def.so -o def.so
# RUN: ld.lld -shared wrap.o def.so -o wrap.so

# RUN: llvm-mc -filetype=obj -triple=x86_64 start.s -o start.o
# RUN: ld.lld --no-allow-shlib-undefined start.o wrap.so ref.so -o /dev/null 2>&1 | count 0

#--- start.s
.globl _start
_start:
  callq wrap_get_foo@PLT

#--- ref.s
.weak foo
.globl ref_get_foo
ref_get_foo:
  movq foo@GOTPCREL(%rip), %rax
  retq

#--- wrap.s
.globl wrap_get_foo
wrap_get_foo:
  movq foo@GOTPCREL(%rip), %rax
  retq

#--- def.s
.data
.globl foo
foo:
  .long 0
