# REQUIRES: x86

# Test that we don't fail with foo being undefined.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --export-dynamic %t.o -o %t

        .weak foo
        .quad foo
