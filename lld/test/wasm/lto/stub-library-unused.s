# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-as %t/unused.ll -o %t/unused.o
# RUN: rm -f %t/libunused.a
# RUN: llvm-ar rcs %t/libunused.a %t/unused.o
# RUN: wasm-ld %t/main.o %t/libunused.a %t/stub.so -o %t.wasm --allow-undefined --why-extract=%t/why.txt
# RUN: obj2yaml %t.wasm | FileCheck %s
# RUN: FileCheck --check-prefix=WHY %s < %t/why.txt

## Test that an unreferenced stub library symbol does not cause its bitcode
## archive dependencies to be extracted or exported during LTO.

# CHECK:        - Name:            _start
# CHECK-NOT:    unused_dep
# CHECK-NOT:    unused_stub

# WHY: reference	extracted	symbol
# WHY-NOT: unused_dep

#--- main.s
.globl _start
_start:
    .functype _start () -> ()
    end_function

#--- unused.ll
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

declare void @unused_stub()

define void @unused_dep() {
entry:
  call void @unused_stub()
  ret void
}

#--- stub.so
#STUB
unused_stub: unused_dep
