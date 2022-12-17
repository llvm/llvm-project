# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This is a regression test. Previously, we would emit a bogus LSDA pointer if
## the following conditions held:
##   * ICF and dead-strip were both done
##   * There exist two functions different compact unwind encodings, but the
##     same LSDA
##
## Essentially, we'd neglected to canonicalize the LSDA pointer after ICF, but
## the broken output would only appear if the compact unwind entry that pointed
## to it was not itself folded.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/test.s -o %t/test.o
# RUN: %lld -dylib -dead_strip --icf=all %t/test.o -o %t/test
# RUN: llvm-objdump --macho --syms --unwind-info %t/test | FileCheck %s

# CHECK:      SYMBOL TABLE:
## Sanity check: Verify that the LSDAs are dedup'ed
# CHECK-NEXT: [[#%.16x, EXC:]]       l     O __TEXT,__gcc_except_tab _exception0
# CHECK-NEXT: [[#EXC]]               l     O __TEXT,__gcc_except_tab _exception1
## But that the functions themselves aren't
# CHECK-NEXT: [[#%.16x, FOO:]]        g     F __TEXT,__text _foo
# CHECK-NEXT: [[#%.16x, BAR:FOO + 1]] g     F __TEXT,__text _bar

## _foo and _bar should share the same LSDA. We would previously emit a bogus
## address for _bar's LSDA.
# CHECK:      Contents of __unwind_info section:
# CHECK:      LSDA descriptors:
# CHECK-NEXT:     [0]: function offset=0x[[#%.8x, FOO]], LSDA offset=0x[[#%.8x, EXC]]
# CHECK-NEXT:     [1]: function offset=0x[[#%.8x, BAR]], LSDA offset=0x[[#%.8x, EXC]]

## But there should be two distinct encodings, one for each function
# CHECK:      Second level indices:
# CHECK-NEXT:     Second level index[0]:
# CHECK-NEXT:       [0]: function offset=0x[[#%.8x, FOO]], encoding[0]=0x42020000
# CHECK-NEXT:       [1]: function offset=0x[[#%.8x, BAR]], encoding[1]=0x42010000

#--- test.s
.text
.globl _foo, _bar

_foo:
  .cfi_startproc
	.cfi_lsda 16, _exception0
	.cfi_def_cfa_offset 16
  ret
  .cfi_endproc

_bar:
  .cfi_startproc
	.cfi_lsda 16, _exception1
	.cfi_def_cfa_offset 8 ## ensure _bar's CUE doesn't get folded with _foo's
  ret
  .cfi_endproc

.section	__TEXT,__gcc_except_tab

_exception0:
  .quad 0x1234

_exception1:
  .quad 0x1234

.subsections_via_symbols
