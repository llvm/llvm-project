# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o /dev/null %t.o -e _not_main
# RUN: not lld -flavor darwinnew -o /dev/null %t.o -e _missing 2>&1 | FileCheck %s
# RUN: not lld -flavor darwinnew -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=DEFAULT_ENTRY

# CHECK: undefined symbol: _missing
# DEFAULT_ENTRY: undefined symbol: _main

.text
.global _not_main
_not_main:
  ret
