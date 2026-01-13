## Test that we properly detect and report out-of-bounds offsets in literal sections.
## We're intentionally testing fatal errors (for malformed input files), and
## fatal errors aren't supported for testing when main is run twice.
# XFAIL: main-run-twice

# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## Test WordLiteralInputSection bounds checking
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/word-literal.s -o %t/word-literal.o
# RUN: not %lld -dylib %t/word-literal.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WORD

## Test CStringInputSection bounds checking
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/cstring.s -o %t/cstring.o
# RUN: not %lld -dylib %t/cstring.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=CSTRING

# WORD: error: {{.*}}word-literal.o:(__literal4): offset is outside the section
# CSTRING: error: {{.*}}cstring.o:(__cstring): offset is outside the section

#--- word-literal.s
.section __TEXT,__literal4,4byte_literals
L_literal:
  .long 0x01020304

.text
.globl _main
_main:
  # We use a subtractor expression to force a section relocation. Symbol relocations
  # don't trigger the error.
  .long L_literal - _main + 4

.subsections_via_symbols

#--- cstring.s
## Create a cstring section with a reference that points past the end
.cstring
L_str:
  .asciz "foo"

.text
.globl _main
_main:
  .long L_str - _main + 4

.subsections_via_symbols