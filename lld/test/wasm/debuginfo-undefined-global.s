# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s

# Debug sections are allowed to contains references to non-live symbols that
# then get GC'd.  In this test the .debug_info section contains a reference to
# foo which is not otherwise used and will not be marked a live in the output.
# Verify the tombstone value is written to debug_info section.

.globaltype foo, i32

.globl  _start
_start:
  .functype _start () -> ()
  end_function

.section .debug_info,"",@
  .int32 foo

foo:

# CHECK:       - Type:            CUSTOM
# CHECK-NEXT:    Name:            .debug_info
# CHECK-NEXT:    Payload:         FFFFFFFF
