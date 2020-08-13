# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo "--- !tapi-tbd-v3" > %t/libinvalidYAML.tbd
# RUN: echo "invalid YAML" >> %t/libinvalidYAML.tbd
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: not lld -flavor darwinnew -Z -L%t -linvalidYAML %t/test.o -o %t/test -Z 2>&1 | FileCheck %s -DDIR=%t

# CHECK: could not load TAPI file at [[DIR]]{{[\\/]}}libinvalidYAML.tbd: malformed file

.globl _main
_main:
  ret
