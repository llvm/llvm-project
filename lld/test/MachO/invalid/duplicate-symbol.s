# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t-dup.o
# RUN: not %lld -dylib -o /dev/null %t-dup.o %t.o 2>&1 | FileCheck %s -DNAME=__ZN1a1bL3fooE -DFILE_1=%t-dup.o -DFILE_2=%t.o
# RUN: not %lld -dylib -o /dev/null %t.o %t.o 2>&1 | FileCheck %s -DNAME=__ZN1a1bL3fooE -DFILE_1=%t.o -DFILE_2=%t.o

# RUN: not %lld -dylib -demangle -o /dev/null %t-dup.o %t.o 2>&1 | FileCheck %s -DNAME="a::b::foo" -DFILE_1=%t-dup.o -DFILE_2=%t.o
# RUN: not %lld -dylib -demangle -o /dev/null %t.o %t.o 2>&1 | FileCheck %s -DNAME="a::b::foo" -DFILE_1=%t.o -DFILE_2=%t.o

# CHECK:      error: duplicate symbol: [[NAME]]
# CHECK-NEXT: >>> defined in [[FILE_1]]
# CHECK-NEXT: >>> defined in [[FILE_2]]

.text
.global __ZN1a1bL3fooE
__ZN1a1bL3fooE:
  ret
