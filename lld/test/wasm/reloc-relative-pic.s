# RUN: sed -e '/^\.globl target/d' -e '/^\.hidden target/d' %s | llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.local32.o
# RUN: sed -e '/^\.globl target/d' -e '/^\.hidden target/d' %s | llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.local64.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.hidden32.o
# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown %s -o %t.hidden64.o
# RUN: sed '/^\.hidden target/d' %s | llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.global32.o
# RUN: sed '/^\.hidden target/d' %s | llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.global64.o
# RUN: wasm-ld -shared --experimental-pic --no-gc-sections %t.local32.o -o %t.local32.wasm
# RUN: wasm-ld -shared --experimental-pic --no-gc-sections -mwasm64 %t.local64.o -o %t.local64.wasm
# RUN: wasm-ld -shared --experimental-pic --no-gc-sections %t.hidden32.o -o %t.hidden32.wasm
# RUN: wasm-ld -shared --experimental-pic --no-gc-sections -mwasm64 %t.hidden64.o -o %t.hidden64.wasm
# RUN: wasm-ld -shared -Bsymbolic --experimental-pic --no-gc-sections %t.global32.o -o %t.global32.wasm
# RUN: wasm-ld -shared -Bsymbolic --experimental-pic --no-gc-sections -mwasm64 %t.global64.o -o %t.global64.wasm
# RUN: wasm-ld -pie --no-entry --experimental-pic --no-gc-sections %t.global32.o -o %t.pie32.wasm
# RUN: wasm-ld -pie --no-entry --experimental-pic --no-gc-sections -mwasm64 %t.global64.o -o %t.pie64.wasm
# RUN: obj2yaml %t.local32.wasm | FileCheck %s
# RUN: obj2yaml %t.local64.wasm | FileCheck %s
# RUN: obj2yaml %t.hidden32.wasm | FileCheck %s
# RUN: obj2yaml %t.hidden64.wasm | FileCheck %s
# RUN: obj2yaml %t.global32.wasm | FileCheck %s
# RUN: obj2yaml %t.global64.wasm | FileCheck %s
# RUN: obj2yaml %t.pie32.wasm | FileCheck %s
# RUN: obj2yaml %t.pie64.wasm | FileCheck %s
# RUN: llvm-objdump -d %t.hidden32.wasm | FileCheck %s --check-prefix=NO-RUNTIME
# RUN: llvm-objdump -d %t.hidden64.wasm | FileCheck %s --check-prefix=NO-RUNTIME

## A preemptible target may live in another module, so its load base does not
## necessarily cancel. Keep rejecting unsupported runtime relocations.
# RUN: not wasm-ld -shared --experimental-pic --no-gc-sections %t.global32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: not wasm-ld -shared --experimental-pic --no-gc-sections -mwasm64 %t.global64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR

## Absolute symbols do not share the module's load base either.
# RUN: sed 's/target-relative/__wasm_first_page_end-relative/g' %s | llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.absolute32.o
# RUN: sed 's/target-relative/__wasm_first_page_end-relative/g' %s | llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.absolute64.o
# RUN: not wasm-ld -shared --experimental-pic --no-gc-sections %t.absolute32.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: not wasm-ld -shared --experimental-pic --no-gc-sections -mwasm64 %t.absolute64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR

.section .data.target,"",@
.globl target
.hidden target
target:
  .int32 42
.size target, 4

.section .data.relative,"",@
.globl relative32
relative32:
  .int32 target-relative32
.size relative32, 4
.globl relative64
relative64:
  .int64 target-relative64
.size relative64, 8
.globl relative32_addend
relative32_addend:
  .int32 target-relative32_addend+8
.size relative32_addend, 4
.globl relative64_addend
relative64_addend:
  .int64 target-relative64_addend+16
.size relative64_addend, 8

## The target and all four references share the same load base. Their differences
## (-4 and -8), including nonzero addends, are resolved statically for both
## wasm32 and wasm64.
# CHECK:      - Type:            DATA
# CHECK:        Content:         {{'?}}2A000000FCFFFFFFF8FFFFFFFFFFFFFFF8FFFFFFFCFFFFFFFFFFFFFF{{'?}}
# NO-RUNTIME-NOT: i32.store
# NO-RUNTIME-NOT: i64.store
# ERROR: invalid runtime relocation type in data section: R_WASM_MEMORY_ADDR_LOCREL_I32
# ERROR: invalid runtime relocation type in data section: R_WASM_MEMORY_ADDR_LOCREL_I64
