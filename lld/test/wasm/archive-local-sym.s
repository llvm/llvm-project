## Test that local symbols in archive files are ignored.
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: rm -f %t/libfoo.a
# RUN: llvm-ar rcs %t/libfoo.a %t/foo.o
# RUN: not wasm-ld %t/libfoo.a %t/main.o -o out.wasm 2>&1 | FileCheck %s

#--- main.s

.functype foo () -> ()

.globl _start
_start:
  .functype _start () -> ()
  call foo
# CHECK: main.o: undefined symbol: foo
  end_function

#--- foo.s

foo:
  .functype foo () -> ()
  end_function
