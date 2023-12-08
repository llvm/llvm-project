# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s
#
# If any table is defined or declared besides the __indirect_function_table,
# the compilation unit should be compiled with -mattr=+reference-types,
# causing symbol table entries to be emitted for all tables.
# RUN: not wasm-ld --no-entry %t.o -o %t.wasm 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

.global call_indirect
call_indirect:
        .functype call_indirect () -> ()
	i32.const 1
        call_indirect () -> ()
        end_function

.globl table
table:
        .tabletype table, externref

# CHECK-ERR: expected one symbol table entry for each of the 2 table(s) present, but got 1 symbol(s) instead.
