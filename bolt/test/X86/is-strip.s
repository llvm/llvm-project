# This test checks whether a binary is stripped or not.

# RUN: %clang++ %cflags %p/Inputs/linenumber.cpp -o %t -Wl,-q
# RUN: llvm-bolt %t -o %t.out 2>&1 | FileCheck %s -check-prefix=CHECK-NOSTRIP
# RUN: cp %t %t.stripped
# RUN: llvm-strip -s %t.stripped
# RUN: not llvm-bolt %t.stripped -o /dev/null 2>&1 | FileCheck %s -check-prefix=CHECK-STRIP
# RUN: llvm-bolt %t.stripped -o %t.out --allow-stripped 2>&1 | FileCheck %s -check-prefix=CHECK-NOSTRIP

# CHECK-NOSTRIP-NOT: BOLT-ERROR: stripped binaries are not supported.
# CHECK-STRIP: BOLT-ERROR: stripped binaries are not supported.
