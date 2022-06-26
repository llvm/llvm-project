# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo "ENTRY(_label)" > %t.script
# RUN: ld.lld %t.script %t.o -o %t
# RUN: llvm-readelf -hs %t | FileCheck %s

# CHECK: Entry point address: 0x[[#%x,ENTRY:]]
# CHECK: [[#]]: {{0*}}[[#ENTRY]]   0 NOTYPE  GLOBAL DEFAULT [[#]] _label

## -e overrides the ENTRY command.
# RUN: echo "ENTRY(_label)" > %t.script
# RUN: ld.lld -e _start -o %t2 %t.script %t.o
# RUN: llvm-readobj --file-headers --symbols %t2 | \
# RUN:   FileCheck -check-prefix=OVERLOAD %s

# OVERLOAD: Entry: [[ENTRY:0x[0-9A-F]+]]
# OVERLOAD: Name: _start
# OVERLOAD-NEXT: Value: [[ENTRY]]

## The entry symbol can be a linker-script-defined symbol.
# RUN: echo "ENTRY(foo); foo = 1;" > %t.script
# RUN: ld.lld -o %t %t.script %t.o
# RUN: llvm-readobj --file-headers --symbols %t | FileCheck --check-prefix=SCRIPT %s

# SCRIPT: Entry: 0x1

## An undefined entry symbol causes a warning.
# RUN: echo "ENTRY(no_such_symbol);" > %t.script
# RUN: ld.lld %t.script %t.o -o /dev/null 2>&1 | FileCheck --check-prefix=MISSING %s

# MISSING: warning: cannot find entry symbol no_such_symbol

.globl _start, _label
_start:
  ret
_label:
  ret
