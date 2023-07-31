// RUN: %clang -miamcu -rtlib=platform %s -### -o %t.o 2>&1 | FileCheck %s

// CHECK: error: the clang compiler does not support 'C++ for IAMCU'
