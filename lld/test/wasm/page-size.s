# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o

        .globl  _start
_start:
        .functype _start () -> (i32)
        i32.const __wasm_first_page_end
        end_function

# Add a symbol to smoke test that `__wasm_first_page_end` is absolute and not
# relative to other data.
        .section .data.foo,"",@
foo:
        .int32  0x11111111
        .size   foo, 4

# RUN: wasm-ld -no-gc-sections -o %t.custom.wasm %t.o --page-size=1
# RUN: obj2yaml %t.custom.wasm | FileCheck %s --check-prefix=CHECK-CUSTOM

# CHECK-CUSTOM:      - Type:            MEMORY
# CHECK-CUSTOM-NEXT:   Memories:
# CHECK-CUSTOM-NEXT:   - Flags:           [ HAS_PAGE_SIZE ]
# CHECK-CUSTOM-NEXT:     Minimum:         0x10410
# CHECK-CUSTOM-NEXT:     PageSize:        0x1

# RUN: llvm-objdump --disassemble-symbols=_start %t.custom.wasm | FileCheck %s --check-prefix=CHECK-CUSTOM-DIS

# CHECK-CUSTOM-DIS:      <_start>:
# CHECK-CUSTOM-DIS:          i32.const 1
# CHECK-CUSTOM-DIS-NEXT:     end

# RUN: wasm-ld -no-gc-sections -o %t.default.wasm %t.o
# RUN: obj2yaml %t.default.wasm | FileCheck %s --check-prefix=CHECK-DEFAULT

# CHECK-DEFAULT:      - Type:            MEMORY
# CHECK-DEFAULT-NEXT:   Memories:
# CHECK-DEFAULT-NEXT:     Minimum:         0x2
# CHECK-DEFAULT-NEXT: - Type:            GLOBAL

# RUN: llvm-objdump --disassemble-symbols=_start %t.default.wasm | FileCheck %s --check-prefix=CHECK-DEFAULT-DIS

# CHECK-DEFAULT-DIS:      <_start>:
# CHECK-DEFAULT-DIS:          i32.const 65536
# CHECK-DEFAULT-DIS-NEXT:     end

# RUN: wasm-ld -no-gc-sections -o %t.custom-import.wasm %t.o --page-size=1 --import-memory
# RUN: obj2yaml %t.custom-import.wasm | FileCheck %s --check-prefix=CHECK-CUSTOM-IMPORT

# CHECK-CUSTOM-IMPORT:      Imports:
# CHECK-CUSTOM-IMPORT-NEXT:   - Module:          env
# CHECK-CUSTOM-IMPORT-NEXT:     Field:           memory
# CHECK-CUSTOM-IMPORT-NEXT:     Kind:            MEMORY
# CHECK-CUSTOM-IMPORT-NEXT:     Memory:
# CHECK-CUSTOM-IMPORT-NEXT:       Flags:           [ HAS_PAGE_SIZE ]
# CHECK-CUSTOM-IMPORT-NEXT:       Minimum:         0x10410
# CHECK-CUSTOM-IMPORT-NEXT:       PageSize:        0x1

# RUN: llvm-objdump --disassemble-symbols=_start %t.custom-import.wasm | FileCheck %s --check-prefix=CHECK-CUSTOM-IMPORT-DIS

# CHECK-CUSTOM-IMPORT-DIS:      <_start>:
# CHECK-CUSTOM-IMPORT-DIS:          i32.const 1
# CHECK-CUSTOM-IMPORT-DIS-NEXT:     end
