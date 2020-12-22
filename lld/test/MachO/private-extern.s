# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/basics.s -o %t/basics.o

## Check that .private_extern symbols are marked as local in the symbol table
## and aren't in the export trie.
# RUN: %lld -dylib %t/basics.o -o %t/basics
# RUN: llvm-objdump --syms --exports-trie %t/basics | \
# RUN:     FileCheck --check-prefix=EXPORTS %s
# RUN: llvm-nm -m %t/basics | FileCheck --check-prefix=EXPORTS-NM %s
# EXPORTS-LABEL: SYMBOL TABLE:
# EXPORTS-DAG:   [[#%x, FOO_ADDR:]] l {{.*}} _foo
# EXPORTS-DAG:   [[#%x, BAR_ADDR:]] g {{.*}} _bar
# EXPORTS-LABEL: Exports trie:
# EXPORTS-NOT:   0x{{0*}}[[#%X, FOO_ADDR]] _foo
# EXPORTS-DAG:   0x{{0*}}[[#%X, BAR_ADDR]] _bar
# EXPORTS-NOT:   0x{{0*}}[[#%X, FOO_ADDR]] _foo
# EXPORTS-NM-DAG: (__TEXT,__cstring) non-external (was a private external) _foo
# EXPORTS-NM-DAG: (__TEXT,__cstring) external _bar

#--- basics.s
.section __TEXT,__cstring

.globl _foo, _bar
.private_extern _foo

_foo:
.asciz "Foo"

_bar:
.asciz "Bar"

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/strong-globl.s -o %t/strong-globl.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-globl.s -o %t/weak-globl.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/strong-private.s -o %t/strong-private.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/weak-private.s -o %t/weak-private.o

## weak + strong symbol takes privateness from strong symbol
## - weak private extern + strong extern = strong extern (for both .o orderings)
# RUN: %lld -dylib %t/weak-private.o %t/strong-globl.o -o %t/wpsg
# RUN: llvm-nm -m %t/wpsg | FileCheck --check-prefix=EXTERNAL %s
# RUN: %lld -dylib %t/strong-globl.o %t/weak-private.o -o %t/sgwp
# RUN: llvm-nm -m %t/sgwp | FileCheck --check-prefix=EXTERNAL %s
# EXTERNAL: (__TEXT,__text) external _foo
## - weak extern + strong private extern = strong private extern
##   (for both .o orderings)
# RUN: %lld -dylib %t/weak-globl.o %t/strong-private.o -o %t/wgsp
# RUN: llvm-nm -m %t/wgsp | FileCheck --check-prefix=NONEXTERNAL %s
# RUN: %lld -dylib %t/strong-private.o %t/weak-globl.o -o %t/spwg
# RUN: llvm-nm -m %t/spwg | FileCheck --check-prefix=NONEXTERNAL %s
# NONEXTERNAL: (__TEXT,__text) non-external (was a private external) _foo

## weak + weak symbol take weaker privateness
## - weak extern + weak private extern = weak extern (both orders)
# RUN: %lld -dylib %t/weak-private.o %t/weak-globl.o -o %t/wpwg
# RUN: llvm-nm -m %t/wpwg | FileCheck --check-prefix=WEAK-EXTERNAL %s
# RUN: %lld -dylib %t/weak-globl.o %t/weak-private.o -o %t/wgwp
# RUN: llvm-nm -m %t/wgwp | FileCheck --check-prefix=WEAK-EXTERNAL %s
# WEAK-EXTERNAL: (__TEXT,__text) weak external _foo
## - weak private extern + weak private extern = weak private extern
# RUN: %lld -dylib %t/weak-private.o %t/weak-private.o -o %t/wpwp
# RUN: llvm-nm -m %t/wpwp | FileCheck --check-prefix=NONEXTERNAL %s

#--- strong-globl.s
.globl _foo
_foo:
  retq

#--- weak-globl.s
.globl _foo
.weak_definition _foo
_foo:
  retq

#--- strong-private.s
.private_extern _foo
.globl _foo
_foo:
  retq

#--- weak-private.s
.private_extern _foo
.globl _foo
.weak_definition _foo
_foo:
  retq

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/comm-small.s -o %t/comm-small.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/comm-large.s -o %t/comm-large.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/comm-small-private.s -o %t/comm-small-private.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     %t/comm-large-private.s -o %t/comm-large-private.o

## For common symbols the larger one wins.
## - smaller private extern + larger extern = larger extern
# RUN: %lld -dylib %t/comm-small-private.o %t/comm-large.o -o %t/cspcl
# RUN: llvm-nm -m %t/cspcl | FileCheck --check-prefix=COMMON-EXTERNAL %s
# RUN: %lld -dylib %t/comm-large.o %t/comm-small-private.o -o %t/clcsp
# RUN: llvm-nm -m %t/clcsp | FileCheck --check-prefix=COMMON-EXTERNAL %s
# COMMON-EXTERNAL: (__DATA,__common) external _foo
## - smaller extern + larger private extern = larger private extern
# RUN: %lld -dylib %t/comm-large-private.o %t/comm-small.o -o %t/clpcs
# RUN: llvm-nm -m %t/clpcs | FileCheck --check-prefix=COMMON-NONEXTERNAL %s
# RUN: %lld -dylib %t/comm-small.o %t/comm-large-private.o -o %t/csclp
# RUN: llvm-nm -m %t/csclp | FileCheck --check-prefix=COMMON-NONEXTERNAL %s
# COMMON-NONEXTERNAL: (__DATA,__common) non-external (was a private external) _foo

# For common symbols with the same size, the privateness of the symbol seen
# later wins (!).
## - equal private extern + equal extern = equal extern (both orders)
# RUN: %lld -dylib %t/comm-small-private.o %t/comm-small.o -o %t/cspcs
# RUN: llvm-nm -m %t/cspcs | FileCheck --check-prefix=COMMON-EXTERNAL %s
## - equal extern + equal private extern = equal private extern (both orders)
# RUN: %lld -dylib %t/comm-small.o %t/comm-small-private.o -o %t/cscsp
# RUN: llvm-nm -m %t/cscsp | FileCheck --check-prefix=COMMON-NONEXTERNAL %s
## - equal private extern + equal private extern = equal private extern
# RUN: %lld -dylib %t/comm-small-private.o %t/comm-small-private.o -o %t/cspcsp
# RUN: llvm-nm -m %t/cspcsp | FileCheck --check-prefix=COMMON-NONEXTERNAL %s

#--- comm-small.s
.comm _foo,4,2

#--- comm-large.s
.comm _foo,8,3

#--- comm-small-private.s
.private_extern _foo
.comm _foo,4,2

#--- comm-large-private.s
.private_extern _foo
.comm _foo,8,3
