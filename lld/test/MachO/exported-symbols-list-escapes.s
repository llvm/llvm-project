# REQUIRES: x86

## An -exported_symbols_list entry may escape glob metacharacters with a
## backslash to name a symbol literally. This matters for Objective-C direct
## methods, whose names contain square brackets that would otherwise be parsed
## as a character class and match nothing.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/bracket.s -o %t/bracket.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/anchor.s -o %t/anchor.o
# RUN: llvm-ar --format=darwin rcs %t/bracket.a %t/bracket.o

## An escaped entry is a literal: it exact-matches, and because literals seed
## addUndefined it also force-loads the archive member that defines it.
# RUN: %lld -dylib -exported_symbols_list %t/escaped.txt \
# RUN:     %t/anchor.o %t/bracket.a -o %t/escaped.dylib
# RUN: llvm-nm --extern-only %t/escaped.dylib | FileCheck --check-prefix=ESCAPED %s

# ESCAPED: -[C m]D

## An unescaped entry keeps its glob meaning: "[C m]" is a character class, so
## it does not match the real symbol and nothing is force-loaded.
# RUN: %lld -dylib -exported_symbols_list %t/unescaped.txt \
# RUN:     %t/anchor.o %t/bracket.a -o %t/unescaped.dylib
# RUN: llvm-nm --extern-only %t/unescaped.dylib | \
# RUN:     FileCheck --check-prefix=UNESCAPED --allow-empty %s

# UNESCAPED-NOT: -[C m]D

## Escaping a star names a symbol that literally contains one, rather than
## matching any symbol by prefix.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/star.s -o %t/star.o
# RUN: %lld -dylib -exported_symbols_list %t/escaped-star.txt \
# RUN:     %t/star.o -o %t/star.dylib
# RUN: llvm-nm --extern-only %t/star.dylib | FileCheck --check-prefix=STAR %s

# STAR-DAG: _lit*eral
# STAR-NOT: _litoteral

#--- bracket.s
.globl "_-[C m]D"
"_-[C m]D":
  retq

#--- anchor.s
.globl _anchor
_anchor:
  retq

#--- star.s
.globl "_lit*eral"
"_lit*eral":
  retq
.globl _litoteral
_litoteral:
  retq

#--- escaped.txt
_-\[C m\]D

#--- unescaped.txt
_-[C m]D

#--- escaped-star.txt
_lit\*eral
