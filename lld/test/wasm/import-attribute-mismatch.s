# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+reference-types -mattr=+exception-handling -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -mattr=+reference-types -mattr=+exception-handling -o %t/other.o %t/other.s
# RUN: not wasm-ld --export call_foo --allow-undefined -o %t.wasm %t/main.o %t/other.o 2>&1 | FileCheck %s

#--- main.s

.functype foo () -> ()
.tabletype mytable, funcref
.globaltype myglobal, i32
.globaltype myglobal, i32
.tagtype mytag i32

.globl _start
_start:
  .functype _start () -> ()
  i32.const foo
  call_indirect mytable, () -> ()
  global.get myglobal
  throw mytag
  end_function

#--- other.s

.functype foo () -> ()
.globaltype myglobal, i32
.tabletype mytable, funcref
.tagtype mytag i32

.globl call_foo
call_foo:
  .functype call_foo () -> ()
  i32.const foo
  call_indirect mytable, () -> ()
  global.get myglobal
  throw mytag
  end_function

.import_module foo, mod1
.import_module mytable, mod2
.import_module myglobal, mod3
.import_module mytag, mod4

# CHECK: wasm-ld: error: import module mismatch for symbol: foo
# CHECK-NEXT: >>> defined as env in {{.*}}main.o
# CHECK-NEXT: >>> defined as mod1 in {{.*}}other.o

# CHECK: wasm-ld: error: import module mismatch for symbol: mytable
# CHECK-NEXT: >>> defined as env in {{.*}}main.o
# CHECK-NEXT: >>> defined as mod2 in {{.*}}other.o

# CHECK: wasm-ld: error: import module mismatch for symbol: myglobal
# CHECK-NEXT: >>> defined as env in {{.*}}main.o
# CHECK-NEXT: >>> defined as mod3 in {{.*}}other.o

# CHECK: wasm-ld: error: import module mismatch for symbol: mytag
# CHECK-NEXT: >>> defined as env in {{.*}}main.o
# CHECK-NEXT: >>> defined as mod4 in {{.*}}other.o
