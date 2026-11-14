## Check that a short-range PC-relative reference in a function that BOLT does
## not emit is still updated when its target is a symbol without a function,
## such as __hot_data_start. Such a target cannot be kept in place, so the
## instruction must be patched as usual: correctly when the new value is in
## range, and with an error otherwise, never silently left pointing at the old
## address. Both LDR (literal) and ADR are covered.
##
## https://github.com/llvm/llvm-project/pull/221344

# REQUIRES: system-linux

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/main.s \
# RUN:   -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q \
# RUN:   -Wl,--no-relax -Wl,-z,max-page-size=4096 -Wl,-T,%t/layout.ld

## Place the new code close to the input so that the relocated
## __hot_data_start stays within the +/-1MB reach of LDR (literal) and ADR.
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=victim --lite=0 --hot-data \
# RUN:   --reorder-data=.data --data=%t/profile.fdata --no-huge-pages \
# RUN:   --custom-allocation-vma=0x80000 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -t -d --disassemble-symbols=victim %t.bolt \
# RUN:   | FileCheck %s

# CHECK-BOLT-NOT: unable to update PC-relative reference to __hot_data_start
# CHECK-BOLT-NOT: unable to convert ADR
# CHECK-BOLT-NOT: BOLT-ERROR

## All references to __hot_data_start follow the new hot data start.
# CHECK: [[#%x,HOT:]] g {{.*}} __hot_data_start
# CHECK-LABEL: <victim>:
# CHECK-NEXT: ldr x0, 0x[[#HOT]]
# CHECK-NEXT: adrp x1, 0x[[#HOT]]
# CHECK-NEXT: add x1, x1, #0x0
# CHECK-NEXT: ldr x1, [x1]
# CHECK-NEXT: adr x2, 0x[[#HOT]]

## With huge page alignment the new value is out of range: BOLT must fail
## instead of leaving the old address in place.
# RUN: not llvm-bolt %t.exe -o %t.far.bolt --skip-funcs=victim --lite=0 \
# RUN:   --hot-data --reorder-data=.data --data=%t/profile.fdata 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-FAR

# CHECK-FAR: BOLT-ERROR: JITLink failed: {{.*}} out of range of {{ADRLiteral21|LDRLiteral19}} fixup

#--- main.s
  .text
  .p2align 2
  .globl _start
  .type _start, %function
_start:
  bl hot_fn
  bl victim
  mov x8, #93
  svc #0
  .size _start, .-_start

## All loads refer to the same marker. The ADRP+ADD pair is patched through
## the regular path; the LDR (literal) and the ADR are the ones under test.
  .globl victim
  .type victim, %function
victim:
  ldr x0, __hot_data_start
  adrp x1, __hot_data_start
  add x1, x1, :lo12:__hot_data_start
  ldr x1, [x1]
  adr x2, __hot_data_start
  ldr x2, [x2]
  cmp x0, x1
  ccmp x1, x2, #0, eq
  cset x0, ne
  ret
  .size victim, .-victim

  .globl hot_fn
  .type hot_fn, %function
hot_fn:
  adrp x2, ordinary_data
  add x2, x2, :lo12:ordinary_data
  ldr x3, [x2]
  add x3, x3, #1
  str x3, [x2]
  ret
  .size hot_fn, .-hot_fn

  .data
  .p2align 3
  .globl __hot_data_start
  .type __hot_data_start, %object
__hot_data_start:
  .size __hot_data_start, 0
  .quad 0x11111111

## Keep a data relocation in the moved object so that BOLT does not append a
## synthetic relocation word after it.
  .globl ordinary_data
  .type ordinary_data, %object
ordinary_data:
  .quad ordinary_data
  .size ordinary_data, .-ordinary_data

  .globl __hot_data_end
  .type __hot_data_end, %object
__hot_data_end:
  .size __hot_data_end, 0

#--- layout.ld
ENTRY(_start)
PHDRS
{
  text PT_LOAD FILEHDR PHDRS FLAGS(5);
  data PT_LOAD FLAGS(6);
}
SECTIONS
{
  . = 0x10000 + SIZEOF_HEADERS;
  .text : { *(.text .text.*) } :text
  . = ALIGN(0x1000);
  .data : { *(.data .data.*) } :data
}

#--- profile.fdata
1 _start 0 1 hot_fn 0 0 100
4 hot_fn 8 4 ordinary_data 0 100
