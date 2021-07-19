# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: not wasm-ld -o %t.wasm %t.o 2>&1 | FileCheck %s -check-prefix=UNDEF
# RUN: wasm-ld --allow-undefined -o %t.wasm %t.o
# RUN: not wasm-ld --shared -o %t.wasm %t.o 2>&1 | FileCheck %s -check-prefix=SHARED

.globl  _start
_start:
  .functype _start () -> (i32)
  i32.const 0
  i32.load  data_external
  end_function

.size data_external, 4

# UNDEF: error: {{.*}}undefined-data.s.tmp.o: undefined symbol: data_external
# SHARED: error: {{.*}}undefined-data.s.tmp.o: relocation R_WASM_MEMORY_ADDR_LEB cannot be used against symbol data_external; recompile with -fPIC
