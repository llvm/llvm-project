# REQUIRES: x86

## For a long time, LLD only included those members from thin archives that were actually used
## during linking. However, we need to iterate over all members for -ObjC, check that we don't
## crash when we encounter a missing member.

# RUN: rm -rf %t && mkdir %t && cd %t
# RUN: sed s/SYM/_main/   %s | llvm-mc -filetype=obj -triple=x86_64-apple-macos -o main.o
# RUN: sed s/SYM/_unused/ %s | llvm-mc -filetype=obj -triple=x86_64-apple-macos -o unused.o

# RUN: llvm-ar rcsT unused.a unused.o; rm unused.o
## FIXME: Absolute paths don't end up relativized in the repro file.

# RUN: %no-fatal-warnings-lld %t/main.o %t/unused.a -ObjC -o /dev/null 2>&1 \
# RUN:                      | FileCheck %s --check-prefix=WARN

# RUN: %lld main.o unused.a -ObjC --no-warn-thin-archive-missing-members 2>&1 | count 0

# WARN: warning: {{.*}}unused.a: -ObjC failed to open archive member: 'unused.o'

.text
.globl SYM
SYM:
    ret
