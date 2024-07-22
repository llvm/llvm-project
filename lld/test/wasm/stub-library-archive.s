# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t/foodeps.o %t/foodeps.s
# RUN: rm -f %t/libfoodeps.a
# RUN: llvm-ar rcs %t/libfoodeps.a %t/foodeps.o
# RUN: wasm-ld %t.o %p/Inputs/libstub.so %t/libfoodeps.a -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

#--- main.s

# The function foo is defined in libstub.so but depends on foodep1 and foodep2

# foodep1 and foodep2 a defined libfoodeps.a(foodeps.o) but this function
# depeds on baz which is also defined in libstub.so.

.functype foo () -> ()
.import_name foo, foo_import

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function

.globl bazdep
bazdep:
    .functype bazdep () -> ()
    end_function

#--- foodeps.s

.functype baz () -> ()

.globl foodep1
foodep1:
  .functype foodep1 () -> ()
  call baz
  end_function

.globl foodep2
foodep2:
  .functype foodep2 () -> ()
  end_function

# CHECK:       - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           2
# CHECK-NEXT:      - Name:            bazdep
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           3
# CHECK-NEXT:      - Name:            foodep1
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           4
# CHECK-NEXT:      - Name:            foodep2
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           5
