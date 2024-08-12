# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s
# ERRUND: error: {{.*}}.o: relocation R_WASM_TABLE_INDEX_REL_SLEB cannot be used against an undefined symbol `external_func`

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=report-all 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --warn-unresolved-symbols 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s

# RUN: not wasm-ld --experimental-pic -shared %t.o -o /dev/null  --unresolved-symbols=ignore-all 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s

.globaltype	__table_base, i32, immutable

.functype external_func () -> ()

.globl _start
_start:
    .functype _start () -> ()
    global.get  __table_base
    i32.const external_func@TBREL
    i32.add
    call_indirect () -> ()
    end_function
