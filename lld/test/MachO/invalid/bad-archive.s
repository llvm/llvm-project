# REQUIRES: x86
# RUN: echo "!<arch>" > %t.a
# RUN: echo "foo" >> %t.a
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: not %lld %t.o %t.a -o /dev/null 2>&1 | FileCheck -DFILE=%t.a %s
# RUN: not %lld %t.o -force_load %t.a -o /dev/null 2>&1 | FileCheck -DFILE=%t.a %s
# RUN: not %lld %t.o -ObjC %t.a -o /dev/null 2>&1 | FileCheck -DFILE=%t.a %s
# CHECK: error: [[FILE]]: failed to parse archive: truncated or malformed archive (remaining size of archive too small for next archive member header at offset 8)

.global _main
_main:
  ret
