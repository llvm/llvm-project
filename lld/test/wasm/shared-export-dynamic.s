# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

# By default all `default` symbols should be exported
# RUN: wasm-ld -shared --experimental-pic -o %t.wasm %t.o
# RUN: obj2yaml %t.wasm | FileCheck %s -check-prefix=DEFAULT
# DEFAULT: foo

# Verify that `--no-export-dynamic` works with `-shared`
# RUN: wasm-ld -shared --experimental-pic --no-export-dynamic -o %t2.wasm %t.o
# RUN: obj2yaml %t2.wasm | FileCheck %s -check-prefix=NO-EXPORT
# NO-EXPORT-NOT: foo

.globl foo

foo:
  .functype foo () -> (i32)
  i32.const 0
  end_function
