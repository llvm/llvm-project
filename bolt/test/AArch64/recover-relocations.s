## Recover a function address after code movement and preserve the original
## page address used to access a data object.
## No --emit-relocs: these operands must be reconstructed from instructions.
# RUN: echo target > %t.order
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --no-relax %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --recover-relocations \
# RUN:   --print-relocation-recovery --print-only=_start \
# RUN:   --reorder-functions=user --function-order=%t.order \
# RUN:   2>&1 | FileCheck %s
# RUN: %python %p/../Inputs/check-recovered-addresses.py aarch64 %t.exe %t.bolt
# RUN: llvm-readelf -h %t.bolt | FileCheck %s --check-prefix=ELF
# RUN: not llvm-bolt %t.exe -o %t.bad --recover-relocations \
# RUN:   --relocs=0 2>&1 | FileCheck %s --check-prefix=MODE
# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.reloc
# RUN: not llvm-bolt %t.reloc -o %t.bad --recover-relocations \
# RUN:   2>&1 | FileCheck %s --check-prefix=MODE
# RUN: llvm-bolt %t.reloc -o %t.normal --reorder-functions=user --function-order=%t.order

# CHECK: BOLT-INFO: forcing --jump-tables=move for relocation recovery
# CHECK: after recover-relocations
# CHECK: adrp x0, target
# CHECK: add x0, x0, :lo12:target
# CHECK: adrp x1, {{.*}}__BOLT_zero_addr
# CHECK-COUNT-2: adrp x3, target
# CHECK: add x3, x3, :lo12:target
# CHECK: adrp x4, {{.*}}__BOLT_zero_addr
# CHECK: add x4, x4, #{{(0x)?[0-9a-f]+}}
# CHECK-COUNT-2: adrp x5, {{.*}}__BOLT_zero_addr
# CHECK: add x5, x5, #{{(0x)?[0-9a-f]+}}
# MODE: relocation recovery requires missing static text relocations
# ELF: Entry point address: 0x{{[1-9a-fA-F][0-9a-fA-F]*}}

.text
.globl _start
.type _start, %function
_start:
.cfi_startproc
  adrp x0, target
  add x0, x0, :lo12:target
  adrp x1, object
  ldr x2, [x1, :lo12:object]
  cbz x2, .Lleft
  adrp x3, target
  b .Ljoin
.Lleft:
  adrp x3, target
.Ljoin:
  add x3, x3, :lo12:target

  // A non-ADRP definition on one incoming path makes the join unsafe.
  cbz x2, .Lunknown
  adrp x4, target
  b .Lunknown_join
.Lunknown:
  mov x4, xzr
.Lunknown_join:
  add x4, x4, :lo12:target

  // ADRPs with different pages at a join must both fall back.
  cbz x2, .Lother_page
  adrp x5, target
  b .Lpage_join
.Lother_page:
  adrp x5, other
.Lpage_join:
  add x5, x5, :lo12:target
  ret
.cfi_endproc
.size _start, .-_start

.p2align 12
.globl target
.type target, %function
target:
.cfi_startproc
  ret
.cfi_endproc
.size target, .-target

.p2align 12
.type other, %function
other:
.cfi_startproc
  ret
.cfi_endproc
.size other, .-other

.data
.p2align 12
object:
  .xword 42
