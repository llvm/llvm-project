# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -dylib %t.o -o %t.dylib

# RUN: llvm-objdump --syms --exports-trie %t.dylib | \
# RUN:   FileCheck %s --check-prefix=EXPORTS
# EXPORTS-LABEL: SYMBOL TABLE:
# EXPORTS-DAG:   [[#%x, HELLO_ADDR:]] {{.*}} _hello
# EXPORTS-DAG:   [[#%x, HELLO_WORLD_ADDR:]] {{.*}} _hello_world
# EXPORTS-DAG:   [[#%x, HELLO_ITS_ME_ADDR:]] {{.*}} _hello_its_me
# EXPORTS-DAG:   [[#%x, HELLO_ITS_YOU_ADDR:]] {{.*}} _hello_its_you
# EXPORTS-LABEL: Exports trie:
# EXPORTS-DAG:   0x{{0*}}[[#%X, HELLO_ADDR]] _hello
# EXPORTS-DAG:   0x{{0*}}[[#%X, HELLO_WORLD_ADDR]] _hello_world
# EXPORTS-DAG:   0x{{0*}}[[#%x, HELLO_ITS_ME_ADDR:]] _hello_its_me
# EXPORTS-DAG:   0x{{0*}}[[#%x, HELLO_ITS_YOU_ADDR:]] _hello_its_you

## Check that we are sharing prefixes in the trie.
# RUN: obj2yaml %t.dylib | FileCheck %s
# CHECK-LABEL: ExportTrie:
# CHECK: Name: ''
# CHECK: Name: _hello
# CHECK: Name: _
# CHECK: Name: world
# CHECK: Name: its_
# CHECK: Name: me
# CHECK: Name: you

.section __TEXT,__cstring
.globl _hello, _hello_world, _hello_its_me, _hello_its_you

## Test for when an entire symbol name is a prefix of another.
_hello:
.asciz "Hello!\n"

_hello_world:
.asciz "Hello world!\n"

.data
_hello_its_me:
.asciz "Hello, it's me\n"

_hello_its_you:
.asciz "Hello, it's you\n"
