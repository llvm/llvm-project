# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -dylib %t.o -o /dev/null

## Check that we parse the LOH & match it to its referent sections correctly,
## even when there are other subsections that don't get parsed as regular
## sections. (We would previously segfault.)
## __debug_info is one such section that gets special-case handling.

.text
_foo:

.section __DWARF,__debug_info,regular,debug

## __StaticInit occurs after __debug_info in the input object file, so the
## LOH-matching code will have to "walk" past __debug_info while searching for
## __StaticInit. Thus this verifies that we can skip past __debug_info
## correctly.
.section __TEXT,__StaticInit
L1:  adrp  x1, _foo@PAGE
L2:  ldr   x1, [x1, _foo@PAGEOFF]

.loh AdrpLdr L1, L2
