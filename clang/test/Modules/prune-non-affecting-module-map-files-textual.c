// This test checks that a module map with a textual header can be marked as
// non-affecting.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- X.modulemap
module X { textual header "X.h" }
//--- X.h
typedef int X_int;

//--- Y.modulemap
module Y { textual header "Y.h" }
//--- Y.h
typedef int Y_int;

//--- A.modulemap
module A { header "A.h" export * }
//--- A.h
#include "X.h"

// RUN: %clang_cc1 -fmodules -emit-module %t/A.modulemap -fmodule-name=A -o %t/A0.pcm \
// RUN:   -fmodule-map-file=%t/X.modulemap
// RUN: %clang_cc1 -module-file-info %t/A0.pcm | FileCheck %s --check-prefix=A0 --implicit-check-not=Y.modulemap
// A0: Input file: {{.*}}X.modulemap

// RUN: %clang_cc1 -fmodules -emit-module %t/A.modulemap -fmodule-name=A -o %t/A1.pcm \
// RUN:   -fmodule-map-file=%t/X.modulemap -fmodule-map-file=%t/Y.modulemap
// RUN: %clang_cc1 -module-file-info %t/A0.pcm | FileCheck %s --check-prefix=A1 \
// RUN:   --implicit-check-not=Y.modulemap
// A1: Input file: {{.*}}X.modulemap

// RUN: diff %t/A0.pcm %t/A1.pcm

//--- B.modulemap
module B { header "B.h" export * }
//--- B.h
#include "A.h"
typedef X_int B_int;

// RUN: %clang_cc1 -fmodules -emit-module %t/B.modulemap -fmodule-name=B -o %t/B.pcm \
// RUN:   -fmodule-file=A=%t/A0.pcm \
// RUN:   -fmodule-map-file=%t/A.modulemap -fmodule-map-file=%t/X.modulemap -fmodule-map-file=%t/Y.modulemap
// RUN: %clang_cc1 -module-file-info %t/B.pcm | FileCheck %s --check-prefix=B \
// RUN:   --implicit-check-not=X.modulemap --implicit-check-not=Y.modulemap
// B: Input file: {{.*}}B.modulemap
