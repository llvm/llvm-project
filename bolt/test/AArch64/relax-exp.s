## Check that llvm-bolt handles code size larger than 256MB.
## Additionally, check veneers: no double veneers in lite mode and proper names
## for BOLT-introduced veneers.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --lite=1 --data %t.fdata \
# RUN:   --print-normalized 2>&1 | FileCheck %s --check-prefix=CHECK-BOLT-LITE
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --lite=0 --data %t.fdata \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-bolt %t.exe -o %t.bolt --relax-exp --hot-functions-at-end --lite=0 \
# RUN:   --data %t.fdata | FileCheck %s --check-prefix=CHECK-BOLT-HOT-END
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

## Constant islands at the end of functions foo(), bar(), and _start() make each
## one of them ~112MB in size. Thus the total code size exceeds 300MB.

  .text
  .global foo
  .type foo, %function
foo:
  bl _start
  bl bar
  ret
  .space 0x7000000
  .size foo, .-foo

  .global bar
  .type bar, %function
bar:
  bl foo
  bl _start
  ret
  .space 0x7000000
  .size bar, .-bar

  .global hot
  .type hot, %function
hot:
# FDATA: 0 [unknown] 0 1 hot 0 0 100
  bl foo
  bl bar
  bl _start
  ret
  .size hot, .-hot

## Check that BOLT sees the call to foo, not to its veneer in lite mode.
# CHECK-BOLT-LITE-LABEL: Binary Function "hot"
# CHECK-BOLT-LITE: bl
# CHECK-BOLT-LITE-SAME: {{[[:space:]]foo[[:space:]]}}

# CHECK-BOLT-LITE-NOT: BOLT-INFO: {{.*}} short thunks created
# CHECK-BOLT-LITE:     BOLT-INFO: 3 long thunks created

## Check the number of thunks created in other modes.
# CHECK-BOLT: BOLT-INFO: 4 short thunks created
# CHECK-BOLT: BOLT-INFO: 3 long thunks created

# CHECK-BOLT-HOT-END: BOLT-INFO: 4 short thunks created
# CHECK-BOLT-HOT-END: BOLT-INFO: 2 long thunks created

## Check that correct veneers are used depending on the target proximity.
# CHECK-OUTPUT-LABEL: <hot>:
# CHECK-OUTPUT-NEXT: bl {{.*}} <__AArch64ADRPThunk_foo>
# CHECK-OUTPUT-NEXT: bl {{.*}} <__AArch64Thunk_bar>
# CHECK-OUTPUT-NEXT: bl {{.*}} <_start>

  .global _start
  .type _start, %function
_start:
  bl foo
  bl bar
  bl hot
  ret
  .space 0x7000000
  .size _start, .-_start
