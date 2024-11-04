// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -fforce-check-cxx20-modules-input-files \
// RUN:     %t/a.cppm -emit-module-interface -o %t/a.pcm
//
// RUN: echo "inline int bar = 46;" >> %t/foo.h
// RUN: not %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -fforce-check-cxx20-modules-input-files %t/a.pcm \
// RUN:     -emit-llvm -o -  2>&1 | FileCheck %t/a.cppm -check-prefix=CHECK-HEADER-FAILURE
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -fforce-check-cxx20-modules-input-files \
// RUN:     %t/a.cppm -emit-module-interface -o %t/a.pcm

// RUN: echo "export int var = 43;" >> %t/a.cppm
//
// RUN: not %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -fforce-check-cxx20-modules-input-files %t/a.pcm \
// RUN:     -emit-llvm -o -  2>&1 | FileCheck %t/a.cppm -check-prefix=CHECK-FAILURE

//--- foo.h
inline int foo = 43;

//--- a.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module a;
export using ::foo;

// CHECK-HEADER-FAILURE: fatal error:{{.*}}foo.h' has been modified since the AST file {{.*}}was built
// CHECK-FAILURE: fatal error:{{.*}}a.cppm' has been modified since the AST file {{.*}}was built
