// RUN: not %clang_cc1 -E -dependency-file bla -MT %t/doesnotexist/bla.o -MP -o %t/doesnotexist/bla.o -x c /dev/null 2>&1 | FileCheck %s

// CHECK: error: unable to open output file

// rdar://9286457
