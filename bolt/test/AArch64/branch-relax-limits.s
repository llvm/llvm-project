## Reject non-call branches beyond 128MiB, but allow in-range branches
## and out-of-range tail calls.

# RUN: %clang %cflags -Wa,-L %s -o %t -Wl,-q,-Ttext=0x10000000
# RUN: link_fdata --no-lbr %s %t %t.fdata
# DEFINE: %{profile-split} = --data=%t.fdata --funcs=_start,foo \
# DEFINE:   --split-functions --split-all-cold --hot-functions-at-end --align-text=0x8000000
# RUN: llvm-bolt %t -o %t.bolt %{profile-split} --keep-nops 2>&1 | FileCheck %s
# RUN: llvm-bolt %t -o %t.boundary %{profile-split} --align-functions=32 \
# RUN:   --print-longjmp 2>&1 | FileCheck %s --check-prefix=BOUNDARY

## Disable relocation mode to keep the cold fragment at its original address.
# DEFINE: %{range-split} = --relocs=0 --funcs=foo --split-functions --split-strategy=all
# RUN: not llvm-bolt %t -o %t.f %{range-split} --custom-allocation-vma=0x20000000 \
# RUN:   2>&1 | FileCheck %s --check-prefix=OUT-OF-RANGE
# RUN: llvm-bolt %t -o %t.ok %{range-split} --custom-allocation-vma=0x11000000 \
# RUN:   2>&1 | FileCheck %s --check-prefix=IN-RANGE
# RUN: llvm-bolt %t -o %t.tail.bolt --funcs=tail_calls \
# RUN:   --custom-allocation-vma=0x200000 --print-longjmp 2>&1 | FileCheck %s --check-prefix=TAILCALL

# CHECK: BOLT-INFO: Inserted 3 stubs in the hot area and 0 stubs in the cold area.
# BOUNDARY: b.eq [[BOUNDARY_STUB:\.LStub[0-9]+]]
# BOUNDARY: [[BOUNDARY_STUB]] (1 instructions,
# BOUNDARY: 00000020: b {{\.Ltmp[0-9]+}}
# OUT-OF-RANGE: BOLT-ERROR: Unable to relax non-call branch beyond 128MiB
# IN-RANGE: BOLT-INFO: Inserted 0 stubs in the hot area and 0 stubs in the cold area.
# TAILCALL: b [[STUB:\.LStub[0-9]+]] # TAILCALL
# TAILCALL: [[STUB]] (
# TAILCALL: adrp x16, foo
# TAILCALL-NEXT: add x16, x16, :lo12:foo
# TAILCALL-NEXT: br x16 # UNKNOWN CONTROL FLOW
# TAILCALL: b [[STUB]] # TAILCALL

    .text
    .globl foo, _start, tail_calls
    .type foo, @function
foo:
    .rept 0x3fffe
    nop
    .endr
    b .Lrange_cold
.Lrange_cold:
    ret
    .size foo, .-foo

    .type _start, @function
_start:
# FDATA: 1 _start #_start# 10
    cmp x0, x1
    b.eq .Lcold
    b .LchangeReg
.Lcold:
# FDATA: 1 _start #.Lcold# 0
    ret
.LchangeReg:
# FDATA: 1 _start #.LchangeReg# 10
    bl foo
    add x0, x1, #0
    .rept 0x3fffb
    nop
    .endr
    ret
    .size _start, .-_start

    .type tail_calls, @function
tail_calls:
    cmp x0, x1
    b.eq .Ltail_change
    b foo
.Ltail_change:
    add x0, x1, #0
    b foo
    .size tail_calls, .-tail_calls
