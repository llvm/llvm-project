# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
# RUN: wasm-ld %t.o %p/Inputs/libstub.so -o %t.wasm
# RUN: obj2yaml %t.wasm | FileCheck %s

# When the dependencies are missing the link fails
# RUN: not wasm-ld %t.o %p/Inputs/libstub-missing-dep.so -o %t.wasm 2>&1 | FileCheck --check-prefix=MISSING-DEP %s

# When the dependencies are missing the link fails
# MISSING-DEP: libstub-missing-dep.so: undefined symbol: missing_dep. Required by foo
# MISSING-DEP: libstub-missing-dep.so: undefined symbol: missing_dep2. Required by foo

# The function foo is defined in libstub.so but depend on foodep1 and foodep2
.functype foo () -> ()
.import_name foo, foo_import

.globl foodep1
foodep1:
  .functype foodep1 () -> ()
  end_function

.globl foodep2
foodep2:
  .functype foodep2 () -> ()
  end_function

.globl _start
_start:
    .functype _start () -> ()
    call foo
    end_function

# CHECK:       - Type:            EXPORT
# CHECK-NEXT:    Exports:
# CHECK-NEXT:      - Name:            memory
# CHECK-NEXT:        Kind:            MEMORY
# CHECK-NEXT:        Index:           0
# CHECK-NEXT:      - Name:            foodep1
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           1
# CHECK-NEXT:      - Name:            foodep2
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           2
# CHECK-NEXT:      - Name:            _start
# CHECK-NEXT:        Kind:            FUNCTION
# CHECK-NEXT:        Index:           3
