# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: not %lld -dylib -framework CoreFoundation --icf=all %t/test.o 2>&1 | FileCheck %s
# CHECK: error: {{.*}}test.o: __cfstring contains symbol _uh_oh at misaligned offset

.cstring
L_.str:
  .asciz  "foo"

.section  __DATA,__cfstring
.p2align  3
L__unnamed_cfstring_:
  .quad  ___CFConstantStringClassReference
  .long  1992 ## utf-8
_uh_oh:
  .space  4
  .quad  L_.str
  .quad  3 ## strlen
