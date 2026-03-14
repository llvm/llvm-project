## We're intentionally testing fatal errors (for malformed input files), and
## fatal errors aren't supported for testing when main is run twice.
# XFAIL: main-run-twice

# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/not-terminated.s -o %t/not-terminated.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/relocs.s -o %t/relocs.o

# RUN: not %lld -dylib %t/not-terminated.o 2>&1 | FileCheck %s --check-prefix=TERM
# RUN: not %lld -dylib %t/relocs.o 2>&1 | FileCheck %s --check-prefix=RELOCS

# TERM:   not-terminated.o:(__cstring+0x4): string is not null terminated
# RELOCS: error: {{.*}}relocs.o: __TEXT,__cstring contains relocations, which is unsupported

#--- not-terminated.s
.cstring
.asciz "foo"
.ascii "oh no"

#--- relocs.s
.cstring
_str:
.asciz "foo"
.quad _str
