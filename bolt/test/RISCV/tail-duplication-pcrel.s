// RUN: %clang %cflags64 -Wl,--no-relax -o %t %s
// RUN: link_fdata --no-lbr %s %t %t.fdata
// RUN: llvm-bolt %t --data=%t.fdata --tail-duplication=aggressive \
// RUN:   --tail-duplication-minimum-offset=1 --print-finalized \
// RUN:   --print-only=_start -o %t.out | FileCheck %s

// CHECK:      beqz a0, {{\.Ltmp[0-9]+}}
// CHECK:      beqz a1, {{\.Ltmp[0-9]+}}
// CHECK:      auipc t0, %pcrel_hi(object) # Label: [[DUP:\.Ltmp[0-9]+]]
// CHECK-NEXT: ld t1, %pcrel_lo([[DUP]])(t0)
// CHECK-NEXT: sd t1, %pcrel_lo([[DUP]])(t0)
// CHECK-NEXT: ret
// CHECK:      ret
// CHECK-NOT:  # Label: [[DUP]]
// CHECK:      auipc t0, %pcrel_hi(object) # Label: [[ORIG:\.Ltmp[0-9]+]]
// CHECK-NEXT: ld t1, %pcrel_lo([[ORIG]])(t0)
// CHECK-NEXT: sd t1, %pcrel_lo([[ORIG]])(t0)
// CHECK-NEXT: ret

  .data
  .p2align 3
  .globl object
object:
  .dword 0

  .text
  .p2align 1
  .globl _start
  .type _start, @function
_start:
  beqz a0, .Lspacer
  beqz a1, tail
pred:
// FDATA: 1 _start #pred# 100
  j tail
.Lspacer:
  ret
tail:
// FDATA: 1 _start #tail# 200
.Lpcrel_hi:
  auipc t0, %pcrel_hi(object)
  ld t1, %pcrel_lo(.Lpcrel_hi)(t0)
  sd t1, %pcrel_lo(.Lpcrel_hi)(t0)
  ret
  .size _start, .-_start
