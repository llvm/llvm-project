# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --no-entry --no-gc-sections -o %t.wasm %t.o
# RUN: llvm-objdump -s %t.wasm | FileCheck %s
# RUN: wasm-ld --no-entry --no-gc-sections --relocatable -o %t.rel.o %t.o
# RUN: llvm-objdump -s %t.rel.o | FileCheck %s --check-prefix=RELOC

# The linker synthesizes a "memtag.globals" custom section describing the
# address and size of every live data symbol carrying the memtag flag.
# The payload is: ULEB version (1), ULEB count, then (ULEB address,
# ULEB size) pairs sorted by address. Untagged symbols are not listed.

        .memtag tagged16
        .globl  tagged16
        .section .data.tagged16,"",@
        .p2align 4
tagged16:
        .int32  42
        .size tagged16, 16

        .memtag local16
        .section .data.local16,"",@
        .p2align 4
local16:
        .int32  1
        .int32  2
        .int32  3
        .int32  4
        .size local16, 16

        .memtag tagged32
        .globl  tagged32
        .section .data.tagged32,"",@
        .p2align 4
tagged32:
        .int64  1
        .int64  2
        .int64  3
        .int64  4
        .size tagged32, 32

        .globl  plain
        .section .data.plain,"",@
        .p2align 2
plain:
        .int32  7
        .size plain, 4

# CHECK:      Contents of section memtag.globals:
# CHECK-NEXT: {{^ [0-9a-f]+ }}01038080 04109080 0410a080 0420

# Relocatable output has no final addresses, and PIC output has none either,
# so the descriptor section is omitted.
# RELOC-NOT: Contents of section memtag.globals:
