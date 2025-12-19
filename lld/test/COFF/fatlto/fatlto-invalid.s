# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=i386-pc-win32 %s -o %t
# RUN: env LLD_IN_TEST=1 not lld-link %t /out:/dev/null /fat-lto-objects 2>&1 | FileCheck %s

# CHECK: error:{{.*}} Invalid bitcode signature

.section        .llvm.lto,"ynD"
L_llvm.embedded.object:
        .asciz  "BC\300\3365\000"
