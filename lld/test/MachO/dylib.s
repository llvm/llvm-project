# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: lld -flavor darwinnew -dylib -install_name @executable_path/libfoo.dylib \
# RUN:   %t.o -o %t.dylib
# RUN: llvm-objdump --macho --dylib-id %t.dylib | FileCheck %s
# CHECK: @executable_path/libfoo.dylib

## If we are building a dylib, we shouldn't error out even if we are passed
## a flag for a missing entry symbol (since dylibs don't have entry symbols).
## Also check that we come up with the right install name if one isn't
## specified.
# RUN: lld -flavor darwinnew -dylib %t.o -o %t.defaultInstallName.dylib -e missing_entry
# RUN: obj2yaml %t.defaultInstallName.dylib | FileCheck %s -DOUTPUT=%t.defaultInstallName.dylib --check-prefix=DEFAULT-INSTALL-NAME
# DEFAULT-INSTALL-NAME: [[OUTPUT]]

## Check for the absence of load commands / segments that should not be in a
## dylib.
# RUN: llvm-objdump --macho --all-headers %t.dylib | FileCheck %s --check-prefix=NCHECK
# NCHECK-NOT: cmd LC_LOAD_DYLINKER
# NCHECK-NOT: cmd LC_MAIN
# NCHECK-NOT: segname __PAGEZERO

# RUN: llvm-objdump --syms --exports-trie %t.dylib | \
# RUN:   FileCheck %s --check-prefix=EXPORTS
# EXPORTS-LABEL: SYMBOL TABLE:
# EXPORTS:       [[#%x, HELLO_WORLD_ADDR:]] {{.*}} _hello_world
# EXPORTS-LABEL: Exports trie:
# EXPORTS:       0x{{0*}}[[#%X, HELLO_WORLD_ADDR]] _hello_world

.section __TEXT,__cstring
.globl _hello_world

_hello_world:
.asciz "Hello world!\n"
