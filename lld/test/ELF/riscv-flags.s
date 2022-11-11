# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 -target-abi=lp64d -mattr=+d %s -o %td.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -target-abi=lp64f -mattr=+f %s -o %tf.o
# RUN: not ld.lld %td.o %tf.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR -DF1=%td.o -DF2=%tf.o

# ERR: error: [[F2]]: cannot link object files with different floating-point ABI from [[F1]]
