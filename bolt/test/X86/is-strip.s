# This test checks whether a binary is stripped or not.

# RUN: %clang++ %p/Inputs/linenumber.cpp -o %t -Wl,-q
# RUN: llvm-bolt %t -o %t.out 2>&1 | FileCheck %s -check-prefix=CHECK-NOSTRIP
# RUN: cp %t %t.stripped
# RUN: llvm-strip -s %t.stripped
# RUN: llvm-bolt %t.stripped -o %t.out 2>&1 | FileCheck %s -check-prefix=CHECK-STRIP

# CHECK-NOSTRIP-NOT: BOLT-INFO: input binary is stripped. The support is limited and is considered experimental.
# CHECK-STRIP: BOLT-INFO: input binary is stripped. The support is limited and is considered experimental.
