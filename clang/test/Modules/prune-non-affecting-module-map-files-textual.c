// This test checks that a module map with a textual header can be marked as
// non-affecting.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- A.modulemap
module A { textual header "A.h" }
//--- B.modulemap
module B { header "B.h" export * }
//--- A.h
typedef int A_int;
//--- B.h
#include "A.h"
typedef A_int B_int;

// RUN: %clang_cc1 -fmodules -emit-module %t/A.modulemap -fmodule-name=A -o %t/A.pcm \
// RUN:   -fmodule-map-file=%t/A.modulemap -fmodule-map-file=%t/B.modulemap

// RUN: %clang_cc1 -fmodules -emit-module %t/B.modulemap -fmodule-name=B -o %t/B0.pcm \
// RUN:   -fmodule-map-file=%t/A.modulemap -fmodule-map-file=%t/B.modulemap -fmodule-file=%t/A.pcm

// RUN: %clang_cc1 -fmodules -emit-module %t/B.modulemap -fmodule-name=B -o %t/B1.pcm \
// RUN:                                    -fmodule-map-file=%t/B.modulemap -fmodule-file=%t/A.pcm

// RUN: diff %t/B0.pcm %t/B1.pcm
