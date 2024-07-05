# REQUIRES: x86

## We only save those archive members from thin archives in the repro file that are actually used
## during linking. This means that if we just iterate members, some of them might not exist.
## Test that that we handle missing members correctly and don't assert on an unchecked Error.
# FIXME: Should we eagerly save all members instead?

# RUN: rm -rf %t; mkdir %t
# RUN: sed s/SYM/_main/   %s | llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o
# RUN: sed s/SYM/_unused/ %s | llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/unused.o

# RUN: cd %t; llvm-ar rcsT unused.a unused.o
## FIXME: Absolute paths don't end up relativized in the repro file.

# RUN: %lld %t/main.o %t/unused.a -ObjC --reproduce=%t/repro.tar -o /dev/null
# RUN: tar xf %t/repro.tar -C %t
# RUN: cd %t/repro; %lld @response.txt

.text
.globl SYM
SYM:
    ret
