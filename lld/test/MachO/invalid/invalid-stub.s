# REQUIRES: x86
# RUN: mkdir -p %t/invalidYAML.framework
# RUN: echo "--- !tapi-tbd-v3" > %t/libinvalidYAML.tbd
# RUN: echo "invalid YAML" >> %t/libinvalidYAML.tbd
# RUN: cp %t/libinvalidYAML.tbd %t/invalidYAML.framework/invalidYAML.tbd
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: not %lld -L%t -linvalidYAML %t/test.o -o %t/test 2>&1 | FileCheck %s -DDIR=%t
# RUN: not %lld -F%t -framework invalidYAML %t/test.o -o %t/test 2>&1 | FileCheck %s -DDIR=%t --check-prefix=CHECK-FRAMEWORK

# CHECK: could not load TAPI file at [[DIR]]{{[\\/]}}libinvalidYAML.tbd: unsupported file type  
# CHECK-FRAMEWORK: could not load TAPI file at [[DIR]]{{[\\/]}}invalidYAML.framework{{[\\/]}}invalidYAML.tbd: unsupported file type  
.globl _main
_main:
  ret
