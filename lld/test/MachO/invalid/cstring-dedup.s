# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/not-terminated.s -o %t/not-terminated.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/relocs.s -o %t/relocs.o

# RUN: not %lld -dylib --deduplicate-literals %t/not-terminated.o 2>&1 | FileCheck %s --check-prefix=TERM
# RUN: not %lld -dylib --deduplicate-literals %t/relocs.o 2>&1 | FileCheck %s --check-prefix=RELOCS

# TERM:   not-terminated.o:(__cstring): string is not null terminated
# RELOCS: relocs.o contains relocations in __TEXT,__cstring, so LLD cannot deduplicate literals. Try re-running without --deduplicate-literals.

#--- not-terminated.s
.cstring
.asciz "foo"
.ascii "oh no"

#--- relocs.s
.cstring
_str:
.asciz "foo"
.quad _str
