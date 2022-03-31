# RUN: %clang -c -o %t %s
# RUN: %llvm_jitlink -preload /usr/lib/swift/libswiftCore.dylib %t

# REQUIRES: osx-swift-runtime

# Check that _typeByName is able to find a type registered by metadata in
# the __swift5_types section.

  .section  __TEXT,__text,regular,pure_instructions
  .globl  _main
  .p2align  4, 0x90
_main:
  .cfi_startproc
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq  %rsp, %rbp
  .cfi_def_cfa_register %rbp
  # Constant String for "4Test3FooV"
  movabsq $8018152761824990260, %rdi
  movabsq $-1585267068834392465, %rsi
  callq _$ss11_typeByNameyypXpSgSSF
  xorl  %edi, %edi
  testq %rax, %rax
  sete  %dil
  callq _exit
  .cfi_endproc

  .private_extern _$s4Test3FooVMa
  .globl  _$s4Test3FooVMa
  .p2align  4, 0x90
_$s4Test3FooVMa:
  leaq  _$s4Test3FooVMf+8(%rip), %rax
  xorl  %edx, %edx
  retq

  .section  __TEXT,__const
l___unnamed_1:
  .asciz  "Test"

  .private_extern _$s4TestMXM
  .globl  _$s4TestMXM
  .weak_definition  _$s4TestMXM
  .p2align  2
_$s4TestMXM:
  .long 0
  .long 0
  .long (l___unnamed_1-_$s4TestMXM)-8

l___unnamed_2:
  .asciz  "Foo"

  .private_extern _$s4Test3FooVMn
  .globl  _$s4Test3FooVMn
  .p2align  2
_$s4Test3FooVMn:
  .long 81
  .long (_$s4TestMXM-_$s4Test3FooVMn)-4
  .long (l___unnamed_2-_$s4Test3FooVMn)-8
  .long (_$s4Test3FooVMa-_$s4Test3FooVMn)-12
  .long (_$s4Test3FooVMF-_$s4Test3FooVMn)-16
  .long 0
  .long 2

  .section  __DATA,__const
  .p2align  3
_$s4Test3FooVMf:
  .quad _$sytWV
  .quad 512
  .quad _$s4Test3FooVMn

  .private_extern "_symbolic _____ 4Test3FooV"
  .section  __TEXT,__swift5_typeref,regular,no_dead_strip
  .globl  "_symbolic _____ 4Test3FooV"
  .weak_definition  "_symbolic _____ 4Test3FooV"
  .p2align  1
"_symbolic _____ 4Test3FooV":
  .byte 1
  .long (_$s4Test3FooVMn-"_symbolic _____ 4Test3FooV")-1
  .byte 0

  .section  __TEXT,__swift5_fieldmd,regular,no_dead_strip
  .p2align  2
_$s4Test3FooVMF:
  .long "_symbolic _____ 4Test3FooV"-_$s4Test3FooVMF
  .long 0
  .short  0
  .short  12
  .long 0

  .section  __TEXT,__swift5_types,regular,no_dead_strip
  .p2align  2
l_type_metadata_table:
  .long _$s4Test3FooVMn-l_type_metadata_table
