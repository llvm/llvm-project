# RUN: llvm-mc -filetype=obj -triple=wasm64-unknown-unknown -o %t.o %s

# RUN: not wasm-ld -mwasm64 --experimental-pic -shared %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld -mwasm64 --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=report-all 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld -mwasm64 --experimental-pic -shared %t.o -o /dev/null  --warn-unresolved-symbols 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld -mwasm64 --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=ignore-all 2>&1 | \
# RUN:   FileCheck %s

# RUN: not wasm-ld -mwasm64 --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=import-dynamic 2>&1 | \
# RUN:   FileCheck %s

.globaltype __memory_base, i64, immutable
.globaltype	__table_base, i64, immutable

.functype external_func () -> ()

call_undefined_function:
    .functype call_undefined_function () -> ()
    global.get  __table_base
    i64.const external_func@TBREL
    # CHECK: error: {{.*}}.o: relocation R_WASM_TABLE_INDEX_REL_SLEB64 is not supported against an undefined symbol `external_func`
    i64.add
    i32.wrap_i64 # Remove when table64 is supported
    call_indirect () -> ()
    end_function
    
access_undefined_data:
    .functype access_undefined_data () -> ()
    global.get  __memory_base
    i64.const external_data@MBREL
    # CHECK: error: {{.*}}.o: relocation R_WASM_MEMORY_ADDR_REL_SLEB64 is not supported against an undefined symbol `external_data`
    i64.add
    i64.load 0
    drop
    end_function

.globl _start
_start:
    .functype _start () -> ()
    call call_undefined_function
    call access_undefined_data
    end_function
