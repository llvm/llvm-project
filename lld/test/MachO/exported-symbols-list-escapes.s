# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/bracket.s -o %t/bracket.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t/anchor.s -o %t/anchor.o
# RUN: llvm-ar --format=darwin rcs %t/bracket.a %t/bracket.o

# RUN: %lld -dylib -exported_symbols_list %t/escaped.txt \
# RUN:     %t/anchor.o %t/bracket.a -o %t/escaped.dylib
# RUN: llvm-nm --extern-only %t/escaped.dylib | FileCheck --check-prefix=ESCAPED %s

# ESCAPED: -[C m]D

# RUN: %lld -dylib -exported_symbols_list %t/opener-only.txt \
# RUN:     %t/anchor.o %t/bracket.a -o %t/opener-only.dylib
# RUN: llvm-nm --extern-only %t/opener-only.dylib | \
# RUN:     FileCheck --check-prefix=OPENER-ONLY %s

# OPENER-ONLY: -[C m]D

# RUN: %lld -dylib -exported_symbols_list %t/unescaped.txt \
# RUN:     %t/anchor.o %t/bracket.a -o %t/unescaped.dylib
# RUN: llvm-nm --extern-only %t/unescaped.dylib | \
# RUN:     FileCheck --check-prefix=UNESCAPED --allow-empty %s

# UNESCAPED-NOT: -[C m]D

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

#--- opener-only.txt
_-\[C m]D

#--- unescaped.txt
_-[C m]D

#--- escaped-star.txt
_lit\*eral
