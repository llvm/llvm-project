# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/secondary.s -o %t/secondary.o

## Check that both main.o and secondary.o contain references to the same
## undefined function and that both are correctly reported.
# RUN: not wasm-ld --no-gc-sections %t/main.o %t/secondary.o -o /dev/null 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s
# ERRUND: error: {{.*}}main.o: undefined symbol: undef_func
# ERRUND: error: {{.*}}secondary.o: undefined symbol: undef_func

## report-all is the default one. Check that we get the same error
# RUN: not wasm-ld --no-gc-sections %t/main.o %t/secondary.o -o /dev/null --unresolved-symbols=report-all 2>&1 | \
# RUN:   FileCheck -check-prefix=ERRUND %s

## Error out if unknown option value was set.
# RUN: not wasm-ld %t/main.o -o /dev/null --unresolved-symbols=xxx 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR1 %s
# ERR1: unknown --unresolved-symbols value: xxx
## Check alias.
# RUN: not wasm-ld %t/main.o -o /dev/null --unresolved-symbols xxx 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR1 %s

## Ignore all should not produce error and should not produce
## any imports.  It should create a stub function in the place of the missing
## function symbol.
# RUN: wasm-ld %t/main.o -o %t2.wasm --unresolved-symbols=ignore-all
# RUN: obj2yaml %t2.wasm | FileCheck -check-prefix=IGNORE %s

## --warn-unresolved-symbols should behave the same
# RUN: wasm-ld %t/main.o -o %t2.wasm --warn-unresolved-symbols
# RUN: obj2yaml %t2.wasm | FileCheck -check-prefix=IGNORE %s

# IGNORE-NOT: - Type:            IMPORT
# IGNORE-NOT: - Type:            ELEM
#
#      IGNORE:  - Type:            CODE
# IGNORE-NEXT:    Functions:
# IGNORE-NEXT:      - Index:           0
# IGNORE-NEXT:        Locals:          []
# IGNORE-NEXT:        Body:            000B
# IGNORE-NEXT:      - Index:           1
# IGNORE-NEXT:        Locals:          []
# IGNORE-NEXT:        Body:            1080808080001082808080001083808080001A1A0B
# IGNORE-NEXT:      - Index:           2
# IGNORE-NEXT:        Locals:          []
# IGNORE-NEXT:        Body:            4180808080000F0B
# IGNORE-NEXT:      - Index:           3
# IGNORE-NEXT:        Locals:          []
# IGNORE-NEXT:        Body:            4180808080000F0B
#
#      IGNORE:  - Type:            CUSTOM
# IGNORE-NEXT:    Name:            name
# IGNORE-NEXT:    FunctionNames:
# IGNORE-NEXT:      - Index:           0
# IGNORE-NEXT:        Name:            undefined
# IGNORE-NEXT:      - Index:           1
# IGNORE-NEXT:        Name:            _start
# IGNORE-NEXT:      - Index:           2
# IGNORE-NEXT:        Name:            get_data_addr
# IGNORE-NEXT:      - Index:           3
# IGNORE-NEXT:        Name:            get_func_addr

## --import-undefined should handle unresolved functions symbols
## by importing them but still report errors/warning for missing data symbols.
## `--allow-undefined` should behave like `--import-undefined` +
## `--unresolve-symbols=ignore`
# RUN: wasm-ld %t/main.o -o %t3.wasm --import-undefined --unresolved-symbols=ignore-all
# RUN: obj2yaml %t3.wasm | FileCheck -check-prefix=IMPORT %s
#      IMPORT:  - Type:            IMPORT
# IMPORT-NEXT:    Imports:
# IMPORT-NEXT:      - Module:          env
# IMPORT-NEXT:        Field:           undef_func
# IMPORT-NEXT:        Kind:            FUNCTION
# IMPORT-NEXT:        SigIndex:        0
# IMPORT-NEXT:  - Type:            FUNCTION

## Check that --import-undefined reports unresolved data symbols.
# RUN: not wasm-ld %t/main.o -o %t3.wasm --import-undefined --unresolved-symbols=report-all 2>&1 | FileCheck -check-prefix=IMPORTUNDEFINED %s
# IMPORTUNDEFINED-NOT: error: {{.*}}main.o: undefined symbol: undef_func
# IMPORTUNDEFINED: error: {{.*}}main.o: undefined symbol: undef_data

## Do not report undefines if linking relocatable.
# RUN: wasm-ld -r %t/main.o -o %t4.wasm --unresolved-symbols=report-all
# RUN: llvm-readobj %t4.wasm > /dev/null 2>&1

## import-dynamic should fail due to incompatible relocations.
# RUN: not wasm-ld %t/main.o -o %t5.wasm --experimental-pic --unresolved-symbols=import-dynamic 2>&1 | FileCheck -check-prefix=ERRNOPIC %s
# ERRNOPIC: relocation R_WASM_MEMORY_ADDR_SLEB cannot be used against symbol `undef_data`; recompile with -fPIC
# ERRNOPIC: relocation R_WASM_TABLE_INDEX_SLEB cannot be used against symbol `undef_func`; recompile with -fPIC

#--- main.s

.functype undef_func () -> ()
.functype get_data_addr () -> (i32)
.functype get_func_addr () -> (i32)

.globl _start
_start:
    .functype _start () -> ()
    call undef_func
    call get_data_addr
    call get_func_addr
    drop
    drop
    end_function

.globl get_data_addr
get_data_addr:
    .functype get_data_addr () -> (i32)
    i32.const undef_data
    return
    end_function

.globl get_func_addr
get_func_addr:
    .functype get_func_addr () -> (i32)
    i32.const undef_func
    return
    end_function

#--- secondary.s

.functype undef_func () -> ()
.globl foo
foo:
    .functype foo () -> ()
    call undef_func
    end_function
