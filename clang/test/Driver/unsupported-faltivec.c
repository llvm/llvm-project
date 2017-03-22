// Tests that clang does not crash with invalid architectures in target triples.
//
// RUN: not %clang -target powerpc64le-linux-gnu -faltivec -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOFALTIVEC %s
// CHECK-NOFALTIVEC: error: the clang compiler does not support 'faltivec', please use -maltivec and include altivec.h explicitly
//
// RUN: not %clang -target powerpc64le-linux-gnu -fno-altivec -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOFNOALTIVEC %s
// CHECK-NOFNOALTIVEC: error: the clang compiler does not support 'fno-altivec', please use -mno-altivec

