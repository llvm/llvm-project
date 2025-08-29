# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 foo.s -o foo.o

# RUN: ld.lld -r  foo.o -T script.ld -o foo_mc.o

# RUN: llvm-objcopy --rename-section .text=.com.text foo_mc.o foo_mc.o
# RUN: llvm-objcopy --rename-section .rela.text=.rela.com.text foo_mc.o foo_mc.o

# RUN: ld.lld -r foo_mc.o  -T script.ld -o foo_mc_after.o

#--- foo.s
  .text
  .globl	foo
  .p2align	4
  .type	foo,@function
foo:
  mov $bar, %rax



#--- script.ld
SECTIONS
{
  .rela.text    0 : { *(.rela.text) }
  .text         0 : { *(.text) }
}

