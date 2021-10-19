# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/not-main.s -o %t/not-main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: %lld -lSystem -dylib %t/libfoo.o -o %t/libfoo.dylib

# RUN: %lld -lSystem -o %t/not-main %t/not-main.o -e _not_main
# RUN: llvm-objdump --macho --all-headers --syms %t/not-main | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE
# CHECK:       {{0*}}[[#%x, ENTRY_ADDR:]] {{.*}} __TEXT,__text _not_main

# CHECK:      sectname __text
# CHECK-NEXT: segname __TEXT
## Note: the following checks assume that the entry symbol is right at the
## beginning of __text.
# CHECK-NEXT: addr 0x{{0*}}[[#ENTRY_ADDR]]
# CHECK-NEXT: size
# CHECK-NEXT: offset [[#ENTRYOFF:]]
# CHECK:      cmd  LC_MAIN
# CHECK-NEXT: cmdsize  24
# CHECK-NEXT: entryoff [[#ENTRYOFF]]

# RUN: %lld -lSystem -o %t/dysym-main %t/not-main.o %t/libfoo.dylib -e _dysym_main
# RUN: llvm-objdump --macho --all-headers --indirect-symbols --lazy-bind %t/dysym-main | FileCheck %s --check-prefix=DYSYM -DDYLIB=libfoo

# RUN: %lld -lSystem -o %t/dyn-lookup %t/not-main.o -e _dysym_main -undefined dynamic_lookup
# RUN: llvm-objdump --macho --all-headers --indirect-symbols --lazy-bind %t/dyn-lookup | FileCheck %s --check-prefix=DYSYM -DDYLIB=flat-namespace

# DYSYM-LABEL: Indirect symbols for (__TEXT,__stubs) 1 entries
# DYSYM-NEXT:  address                      index  name
# DYSYM-NEXT:  0x[[#%x,DYSYM_ENTRY_ADDR:]]  [[#]]  _dysym_main
# DYSYM-LABEL: cmd  LC_MAIN
# DYSYM-NEXT:  cmdsize  24
# DYSYM-NEXT:  entryoff [[#%u, DYSYM_ENTRY_ADDR - 0x100000000]]
# DYSYM-LABEL: Lazy bind table:
# DYSYM-NEXT:  segment  section            address     dylib        symbol
# DYSYM-NEXT:  __DATA   __la_symbol_ptr    {{.*}}      [[DYLIB]]    _dysym_main

# RUN: %lld -lSystem -o %t/weak-dysym-main %t/not-main.o %t/libfoo.dylib -e _weak_dysym_main
# RUN: llvm-objdump --macho --all-headers --indirect-symbols --bind --weak-bind %t/weak-dysym-main | FileCheck %s --check-prefix=WEAK-DYSYM
# WEAK-DYSYM-LABEL: Indirect symbols for (__TEXT,__stubs) 1 entries
# WEAK-DYSYM-NEXT:  address                      index  name
# WEAK-DYSYM-NEXT:  0x[[#%x,DYSYM_ENTRY_ADDR:]]  [[#]]  _weak_dysym_main
# WEAK-DYSYM:       cmd  LC_MAIN
# WEAK-DYSYM-NEXT:  cmdsize  24
# WEAK-DYSYM-NEXT:  entryoff [[#%u, DYSYM_ENTRY_ADDR - 0x100000000]]
# WEAK-DYSYM-LABEL: Bind table:
# WEAK-DYSYM-NEXT:  segment  section          address  type     addend  dylib   symbol
# WEAK-DYSYM:       __DATA   __la_symbol_ptr  {{.*}}   pointer       0  libfoo  _weak_dysym_main
# WEAK-DYSYM-LABEL: Weak bind table:
# WEAK-DYSYM-NEXT:  segment  section          address  type     addend  symbol
# WEAK-DYSYM-NEXT:  __DATA   __la_symbol_ptr  {{.*}}   pointer       0  _weak_dysym_main

# RUN: %lld -dylib -lSystem -o %t/dysym-main.dylib %t/not-main.o %t/libfoo.dylib -e _dysym_main
# RUN: llvm-objdump --macho --indirect-symbols --lazy-bind %t/dysym-main.dylib | FileCheck %s --check-prefix=DYLIB-NO-MAIN
# DYLIB-NO-MAIN-NOT: _dysym_main

# RUN: not %lld -o /dev/null %t/not-main.o -e _missing 2>&1 | FileCheck %s --check-prefix=UNDEFINED
# UNDEFINED: error: undefined symbol: _missing
# RUN: not %lld -o /dev/null %t/not-main.o 2>&1 | FileCheck %s --check-prefix=DEFAULT-ENTRY
# DEFAULT-ENTRY:      error: undefined symbol: _main
# DEFAULT-ENTRY-NEXT: >>> referenced by the entry point

#--- libfoo.s
.text
.global _dysym_main, _weak_dysym_main
.weak_definition _weak_dysym_main
_dysym_main:
  ret

_weak_dysym_main:
  ret

#--- not-main.s
.text
.global _not_main
_not_main:
  ret
