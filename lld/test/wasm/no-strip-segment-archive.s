# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/lib.o %t/lib.s
# RUN: rm -f %t/lib.a
# RUN: llvm-ar rcs %t/lib.a %t/lib.o
# RUN: wasm-ld %t/main.o %t/lib.a --no-entry --allow-undefined -o %t/main.wasm
# RUN: obj2yaml %t/main.wasm | FileCheck %s

# A user-defined wasm custom section inside an archive member should survive
# linking even when no symbol in that member is referenced. Custom sections
# (wasm-bindgen / wit-bindgen / coverage metadata, etc.) have no symbol table
# entries by construction, so symbol-driven archive extraction cannot reach
# them. Without explicit handling the archive member is never extracted and
# the custom section is silently dropped.

#--- main.s
  .globl _start
_start:
  .functype _start () -> ()
  end_function

#--- lib.s
  .section .custom_section.my_meta,"",@
  .asciz "CUSTOM_PAYLOAD"

# The custom section should be present in the linked binary.
# CHECK:        - Type:            CUSTOM
# CHECK:          Name:            my_meta
# CHECK:          Payload:         435553544F4D5F5041594C4F414400
