# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=report-all 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --warn-unresolved-symbols 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=ignore-all 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=import-dynamic 2>&1 | \
# RUN:   FileCheck %s

## These errors should not be reported under -r/--relocation (i.e. when
## generating an object file)
# RUN: wasm-ld --experimental-pic -r %t.o -o /dev/null

.functype external_func () -> ()

use_undefined_function:
    .functype use_undefined_function () -> ()
    i32.const external_func@TBREL
    # CHECK: error: {{.*}}.o: relocation R_WASM_TABLE_INDEX_REL_SLEB is not supported against an undefined symbol `external_func`
    drop
    end_function

use_undefined_data:
    .functype use_undefined_data () -> ()
    i32.const external_data@MBREL
    # CHECK: error: {{.*}}.o: relocation R_WASM_MEMORY_ADDR_REL_SLEB is not supported against an undefined symbol `external_data`
    drop
    end_function

.globl _start
_start:
    .functype _start () -> ()
    call use_undefined_function
    call use_undefined_data
    end_function
