# REQUIRES: x86

## If for some bizarre reason the input file defines its own ___dso_handle, we
## should raise an error. At least, we've implemented this behavior if the
## conflicting symbol is a global. A local symbol of the same name will still
## take priority in our implementation, unlike in ld64. But that's a pretty
## far-out edge case that should be safe to ignore.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -dylib %t.o -o %t.dylib 2>&1 | FileCheck %s -DFILE=%t.o
# CHECK: error: found defined symbol from [[FILE]] with illegal name ___dso_handle

.globl _main, ___dso_handle
.text
_main:
  leaq ___dso_handle(%rip), %rdx
  ret

___dso_handle:
  .space 1
