# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o

# RUN: not wasm-ld %t.o -o does_not_exist/output 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=NO-DIR-OUTPUT,CHECK
# RUN: not wasm-ld %t.o -o %s/dir_is_a_file 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=DIR-IS-OUTPUT,CHECK

# RUN: not wasm-ld %t.o -o %t -Map=does_not_exist/output 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=NO-DIR-MAP,CHECK
# RUN: not wasm-ld %t.o -o %t -Map=%s/dir_is_a_file 2>&1 | \
# RUN:   FileCheck %s -check-prefixes=DIR-IS-MAP,CHECK

# NO-DIR-OUTPUT: error: cannot open output file does_not_exist/output:
# DIR-IS-OUTPUT: error: cannot open output file {{.*}}/dir_is_a_file:
# NO-DIR-MAP: error: cannot open map file does_not_exist/output:
# DIR-IS-MAP: error: cannot open map file {{.*}}/dir_is_a_file:

# We should exit before doing the actual link. If an undefined symbol error is
# discovered we haven't bailed out early as expected.
# CHECK-NOT: undefined_symbol

# RUN: not wasm-ld %t.o -o / 2>&1 | FileCheck %s -check-prefixes=ROOT,CHECK
# ROOT: error: cannot open output file /

.functype undefined_symbol () -> ()

_start:
    .functype _start () -> ()
    call undefined_symbol
    end_function
