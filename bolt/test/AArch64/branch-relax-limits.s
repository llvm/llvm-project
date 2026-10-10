## Reject non-call branches beyond 128MiB, but allow in-range branches
## and out-of-range tail calls.

# RUN: %clang %cflags %s -o %t -Wl,-Ttext=0x10000000
# RUN: not llvm-bolt %t -o %t.f --split-functions --split-strategy=all \
# RUN:   --custom-allocation-vma=0x20000000 2>&1 | FileCheck %s
# RUN: llvm-bolt %t -o %t.ok --split-functions --split-strategy=all \
# RUN:   --custom-allocation-vma=0x11000000 2>&1 | FileCheck %s --check-prefix=IN-RANGE
# RUN: %clang %cflags %s -o %t.tail -Wl,-q,-Ttext=0x10000000
# RUN: llvm-bolt %t.tail -o %t.tail.bolt --funcs=foo \
# RUN:   --custom-allocation-vma=0x200000 --print-longjmp 2>&1 | FileCheck %s --check-prefix=TAILCALL

# CHECK: BOLT-ERROR: Unable to relax non-call branch beyond 128MiB
# IN-RANGE: BOLT-INFO: Inserted 0 stubs in the hot area and 0 stubs in the cold area.
# TAILCALL: Binary Function "foo" after long-jmp
# TAILCALL: b .LStub0 # TAILCALL
# TAILCALL: adrp x16, bar
# TAILCALL-NEXT: add x16, x16, :lo12:bar
# TAILCALL-NEXT: br x16 # UNKNOWN CONTROL FLOW

    .text
    .globl _start
    .type _start, @function
_start:
    mov x0, #1
    b .Lcold
.Lcold:
    ret
    .size _start, .-_start

    .globl foo
    .type foo, @function
foo:
    b bar
    .size foo, .-foo

    .globl bar
    .type bar, @function
bar:
    ret
    .size bar, .-bar
