# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o
# RUN: %lld -lSystem -dead_strip -why_live _foo -why_live _undef -U _undef \
# RUN:   -why_live _support -why_live _support_refs_dylib_fun \
# RUN:   -why_live _abs %t.o -o /dev/null 2>&1 | FileCheck %s

## Due to an implementation detail, LLD is not able to report -why_live info for
## absolute symbols. (ld64 has the same shortcoming.)
# CHECK-NOT:   _abs
# CHECK:       _foo from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:    _quux from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:  _undef from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:    _main from {{.*}}why-live.s.tmp.o
## Our handling of live_support sections can be improved... we should print the
## dylib symbol that keeps _support_refs_dylib_fun alive, instead of printing
## the live_support symbol's name itself. (ld64 seems to have the same issue.)
# CHECK-NEXT: _support_refs_dylib_fun from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:   _support_refs_dylib_fun from {{.*}}why-live.s.tmp.o
## Again, this can be improved: we shouldn't be printing _support twice. (ld64
## seems to have the same issue.)
# CHECK-NEXT:  _support from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:    _support from {{.*}}why-live.s.tmp.o
# CHECK-NEXT:      _foo from {{.*}}why-live.s.tmp.o
# CHECK-EMPTY:

.text
_foo:
  retq

_bar:
  retq

_baz:
  callq _foo
  retq

.no_dead_strip _quux
_quux:
  callq _foo
  retq

.globl _main
_main:
  callq _foo
  callq _baz
  callq _undef
  callq ___isnan
  retq

.globl _abs
_abs = 0x1000

.section __TEXT,support,regular,live_support
_support:
  callq _foo
  callq _abs
  retq

_support_refs_dylib_fun:
  callq ___isnan
  retq

.subsections_via_symbols
