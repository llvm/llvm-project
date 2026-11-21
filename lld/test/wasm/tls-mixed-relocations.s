## One object references cell@TLSREL while the other defines cell in ordinary .data.
# RUN: split-file %s %t
# RUN: llvm-mc -triple=wasm32 -filetype=obj %t/data.s -o %t/data.o
# RUN: llvm-mc -triple=wasm32 -filetype=obj %t/ref.s -o %t/ref.o
# RUN: wasm-ld --no-entry --export=offset %t/data.o %t/ref.o -o %t/gc.wasm
# RUN: llvm-objdump -d %t/gc.wasm | FileCheck %s -DOFFSET=1
# RUN: wasm-ld --no-entry --no-gc-sections %t/data.o %t/ref.o -o %t/live.wasm
# RUN: llvm-objdump -d %t/live.wasm | FileCheck %s -DOFFSET=-3

## With .tdata discarded, __tls_base is at cell. With .tdata live, it is four bytes later.
# CHECK-LABEL: <offset>:
# CHECK:       i32.const [[OFFSET]]{{$}}

## With .tdata live, cell@TLSREL-2147483648 is -2147483652.
## This wraps to 2147483644 on wasm32 and keeps its full value on wasm64.
# RUN: wasm-ld --no-entry --no-gc-sections --compress-relocations --strip-debug %t/data.o %t/ref.o -o %t/compressed.wasm
# RUN: llvm-objdump --disassemble-symbols=offset %t/compressed.wasm | FileCheck %s --check-prefix=WASM32
# WASM32: 41 fc ff ff ff 07 {{.*}}i32.const 2147483644

# RUN: llvm-mc -triple=wasm64 -filetype=obj %t/data.s -o %t/data64.o
# RUN: sed 's/i32/i64/g' %t/ref.s | llvm-mc -triple=wasm64 -filetype=obj -o %t/ref64.o
# RUN: wasm-ld -mwasm64 --no-entry --no-gc-sections --compress-relocations --strip-debug %t/data64.o %t/ref64.o -o %t/compressed64.wasm
# RUN: llvm-objdump --disassemble-symbols=offset %t/compressed64.wasm | FileCheck %s --check-prefix=WASM64
# WASM64: i64.const -2147483652{{$}}

#--- ref.s
.globl offset
offset:
  .functype offset () -> (i32)
  i32.const cell@TLSREL-2147483648
  drop
  i32.const cell@TLSREL+1
  end_function

.section .tdata,"T",@
  .int32 0

## The object that references cell was compiled with TLS support.
.section .custom_section.target_features,"",@
  .byte 2
  .byte '+', 7
  .ascii "atomics"
  .byte '+', 11
  .ascii "bulk-memory"

#--- data.s
.section .data,"",@
.globl cell
cell:
  .int32 42
  .size cell, 4
