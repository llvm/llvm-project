# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown -o %t.o %s

.globl  _start
_start:
  .functype _start () -> ()
  end_function

# RUN: not wasm-ld -o %t.exe 2>&1 | FileCheck -check-prefix=IN %s
# IN: error: no input files

# RUN: not wasm-ld %t.o 2>&1 | FileCheck -check-prefix=OUT %s
# OUT: error: no output file specified

# RUN: not wasm-ld 2>&1 | FileCheck -check-prefix=BOTH %s
# BOTH:     error: no input files
# BOTH-NOT: error: no output file specified

# RUN: not wasm-ld --export-table --import-table %t.o 2>&1 \
# RUN:   | FileCheck -check-prefix=TABLE %s
# TABLE: error: --import-table and --export-table may not be used together

# RUN: not wasm-ld --relocatable --shared-memory %t.o 2>&1 \
# RUN:   | FileCheck -check-prefix=SHARED-MEM %s
# SHARED-MEM: error: -r and --shared-memory may not be used together

# RUN: wasm-ld %t.o -z foo -o /dev/null 2>&1 | FileCheck -check-prefix=ERR10 %s
# RUN: wasm-ld %t.o -z foo -o /dev/null --version 2>&1 | FileCheck -check-prefix=ERR10 %s
# ERR10: warning: unknown -z value: foo

## Check we report "unknown -z value" error even with -v.
# RUN: wasm-ld %t.o -z foo -o /dev/null -v 2>&1 | FileCheck -check-prefix=ERR10 %s

## Note: in GNU ld, --fatal-warning still leads to a warning.
# RUN: not wasm-ld %t.o -z foo --fatal-warnings 2>&1 | FileCheck --check-prefix=ERR10-FATAL %s
# ERR10-FATAL: error: unknown -z value: foo

## stack-size without an = is also an error
# RUN: not wasm-ld %t.o -z stack-size 2>&1 | FileCheck -check-prefix=ERR11 %s
# ERR11: unknown -z value: stack-size
