# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -arch x86_64 -o %t %t.o -e _not_main
# RUN: llvm-objdump --macho --all-headers --syms %t | FileCheck %s
# CHECK-LABEL: SYMBOL TABLE
# CHECK-NEXT: {{0*}}[[#%x, ENTRY_ADDR:]] {{.*}} __TEXT,__text _not_main
# CHECK:      cmd  LC_MAIN
# CHECK-NEXT: cmdsize  24
# CHECK-NEXT: entryoff [[#ENTRYOFF:]]
# CHECK:      sectname __text
# CHECK-NEXT: segname __TEXT
## Note: the following checks assume that the entry symbol is right at the
## beginning of __text.
# CHECK-NEXT: addr 0x{{0*}}[[#ENTRY_ADDR]]
# CHECK-NEXT: size
# CHECK-NEXT: offset [[#ENTRYOFF]]


# RUN: not lld -flavor darwinnew -arch x86_64 -o /dev/null %t.o -e _missing 2>&1 | FileCheck %s --check-prefix=UNDEFINED
# UNDEFINED: error: undefined symbol: _missing
# RUN: not lld -flavor darwinnew -arch x86_64 -o /dev/null %t.o 2>&1 | FileCheck %s --check-prefix=DEFAULT-ENTRY
# DEFAULT-ENTRY: error: undefined symbol: _main

.text
.global _not_main
_not_main:
  movq $0, %rax
  retq
