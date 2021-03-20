# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## We are intentionally building an executable here instead of a dylib / bundle
## in order that the `__PAGEZERO` segment is present, which in turn means that
## the image base starts at a non-zero address. This allows us to verify that
## addresses in the export trie are correctly encoded as relative to the image
## base.
# RUN: %lld %t.o -o %t

# RUN: llvm-objdump --syms --exports-trie %t | FileCheck %s --check-prefix=EXPORTS
# EXPORTS-LABEL: SYMBOL TABLE:
# EXPORTS-DAG:   [[#%x, MAIN_ADDR:]] {{.*}} _main
# EXPORTS-DAG:   [[#%x, HELLO_ADDR:]] {{.*}} _hello
# EXPORTS-DAG:   [[#%x, HELLO_WORLD_ADDR:]] {{.*}} _hello_world
# EXPORTS-DAG:   [[#%x, HELLO_ITS_ME_ADDR:]] {{.*}} _hello_its_me
# EXPORTS-DAG:   [[#%x, HELLO_ITS_YOU_ADDR:]] {{.*}} _hello_its_you
# EXPORTS-DAG:   {{0+}} g *ABS* __mh_execute_header
# EXPORTS-LABEL: Exports trie:
# EXPORTS-DAG:   0x{{0+}} __mh_execute_header [absolute]
# EXPORTS-DAG:   0x{{0*}}[[#%X, MAIN_ADDR]] _main
# EXPORTS-DAG:   0x{{0*}}[[#%X, HELLO_ADDR]] _hello
# EXPORTS-DAG:   0x{{0*}}[[#%X, HELLO_WORLD_ADDR]] _hello_world
# EXPORTS-DAG:   0x{{0*}}[[#%x, HELLO_ITS_ME_ADDR:]] _hello_its_me
# EXPORTS-DAG:   0x{{0*}}[[#%x, HELLO_ITS_YOU_ADDR:]] _hello_its_you

## Check that we are sharing prefixes in the trie.
# RUN: obj2yaml %t | FileCheck %s
# CHECK-LABEL: ExportTrie:
# CHECK: Name: ''
# CHECK: Name: _
# CHECK-DAG: Name: _mh_execute_header
# CHECK-DAG: Name: main
# CHECK-DAG: Name: hello
# CHECK: Name: _
# CHECK: Name: world
# CHECK: Name: its_
# CHECK-DAG: Name: you
# CHECK-DAG: Name: me



.section __TEXT,__cstring
.globl _hello, _hello_world, _hello_its_me, _hello_its_you, _main

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

_main:
  ret
