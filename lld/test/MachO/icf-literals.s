# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -lSystem --icf=all -o %t/test %t/test.o
# RUN: llvm-objdump --macho --syms -d %t/test | FileCheck %s

# CHECK:      _main:
# CHECK-NEXT: callq   _foo2_ref
# CHECK-NEXT: callq   _foo2_ref
# CHECK-NEXT: callq   _bar2_ref
# CHECK-NEXT: callq   _bar2_ref
# CHECK-NEXT: callq   _baz2_ref
# CHECK-NEXT: callq   _baz2_ref
# CHECK-NEXT: callq   _qux2_ref
# CHECK-NEXT: callq   _qux2_ref
# CHECK-NEXT: callq   _sub_str_a_b
# CHECK-NEXT: callq   _sub_str_b_a
# CHECK-NEXT: callq   _sub_lit_a_b
# CHECK-NEXT: callq   _sub_lit_b_a

# CHECK:      [[#%.16x,FOO:]]     l     O __TEXT,__cstring _foo1
# CHECK-NEXT: [[#%.16x,FOO:]]     l     O __TEXT,__cstring _foo2
# CHECK-NEXT: [[#%.16x,BAR:]]     l     O __TEXT,__cstring _bar1
# CHECK-NEXT: [[#%.16x,BAR:]]     l     O __TEXT,__cstring _bar2
# CHECK-NEXT: [[#%.16x,BAZ:]]     l     O __TEXT,__literals _baz1
# CHECK-NEXT: [[#%.16x,BAZ:]]     l     O __TEXT,__literals _baz2
# CHECK-NEXT: [[#%.16x,QUX:]]     l     O __TEXT,__literals _qux1
# CHECK-NEXT: [[#%.16x,QUX:]]     l     O __TEXT,__literals _qux2
# CHECK-NEXT: [[#%.16x,FOO_REF:]] l     F __TEXT,__text _foo1_ref
# CHECK-NEXT: [[#%.16x,FOO_REF:]] l     F __TEXT,__text _foo2_ref
# CHECK-NEXT: [[#%.16x,BAR_REF:]] l     F __TEXT,__text _bar1_ref
# CHECK-NEXT: [[#%.16x,BAR_REF:]] l     F __TEXT,__text _bar2_ref
# CHECK-NEXT: [[#%.16x,BAZ_REF:]] l     F __TEXT,__text _baz1_ref
# CHECK-NEXT: [[#%.16x,BAZ_REF:]] l     F __TEXT,__text _baz2_ref
# CHECK-NEXT: [[#%.16x,QUX_REF:]] l     F __TEXT,__text _qux1_ref
# CHECK-NEXT: [[#%.16x,QUX_REF:]] l     F __TEXT,__text _qux2_ref

## _foo1 vs _bar1: same section, different offsets
## _foo1 vs _baz1: same offset, different sections

.cstring
_foo1:
  .asciz "foo"
_foo2:
  .asciz "foo"
_bar1:
  .asciz "bar"
_bar2:
  .asciz "bar"

.literal8
_baz1:
  .quad 0xdead
_baz2:
  .quad 0xdead
_qux1:
  .quad 0xbeef
_qux2:
  .quad 0xbeef

.text
_foo1_ref:
  leaq _foo1(%rip), %rax
_foo2_ref:
  leaq _foo2(%rip), %rax
_bar1_ref:
  leaq _bar1(%rip), %rax
_bar2_ref:
  leaq _bar2(%rip), %rax
_baz1_ref:
  movq _baz1(%rip), %rax
_baz2_ref:
  movq _baz2(%rip), %rax
_qux1_ref:
  movq _qux1(%rip), %rax
_qux2_ref:
  movq _qux2(%rip), %rax

## _sub_str_a_b and _sub_str_b_a should not be folded: They contain relocations
## against the same string symbols, but in a different order and hence
## return different numbers.
_sub_str_a_b:
  leaq _foo2(%rip), %rdx
  leaq _bar2(%rip), %rax
  subq %rdx, %rax
_sub_str_b_a:
  leaq _bar2(%rip), %rdx
  leaq _foo2(%rip), %rax
  subq %rdx, %rax

## Same with literals instead of strings.
_sub_lit_a_b:
  movq _baz2(%rip), %rax
  subq _qux2(%rip), %rax
_sub_lit_b_a:
  movq _qux2(%rip), %rax
  subq _baz2(%rip), %rax


.globl _main
_main:
  callq _foo1_ref
  callq _foo2_ref
  callq _bar1_ref
  callq _bar2_ref
  callq _baz1_ref
  callq _baz2_ref
  callq _qux1_ref
  callq _qux2_ref
  callq _sub_str_a_b
  callq _sub_str_b_a
  callq _sub_lit_a_b
  callq _sub_lit_b_a

.subsections_via_symbols
