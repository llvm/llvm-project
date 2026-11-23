# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: wasm-ld --allow-undefined -o %t.wasm %t.o

# Fails due to undefined 'foo'
# RUN: not wasm-ld --undefined=baz -o %t.wasm %t.o 2>&1 | FileCheck %s
# CHECK: error: {{.*}}.o: undefined symbol: foo
# CHECK-NOT: undefined symbol: baz

# Succeeds if we pass a file containing 'foo' as --allow-undefined-file.
# RUN: echo 'foo' > %t.txt
# RUN: wasm-ld --allow-undefined-file=%t.txt -o %t.wasm %t.o

# Succeeds even if a missing symbol is added via --export
# RUN: wasm-ld --allow-undefined --export=xxx -o %t.wasm %t.o

.globl _start
_start:
    .functype _start () -> ()
    i32.const bar
    drop
    end_function

.functype foo () -> (i32)

.section .data.bar,"",@
.globl bar
bar:
    # Takes the address of the external foo() resulting in undefined external
    .int32 foo
    .size bar, 4
