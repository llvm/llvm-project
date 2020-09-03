# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld -o %t.wasm %t.o 2>&1 | FileCheck %s

# CHECK: error: {{.*}}.o: undefined symbol: foo(int)

# RUN: not wasm-ld --no-demangle \
# RUN:     -o %t.wasm %t.o 2>&1 | FileCheck -check-prefix=CHECK-NODEMANGLE %s

# CHECK-NODEMANGLE: error: {{.*}}.o: undefined symbol: _Z3fooi

  .globl  _start
_start:
  .functype _start () -> ()
  i32.const 1
  call  _Z3fooi
  end_function

.functype _Z3fooi (i32) -> ()
