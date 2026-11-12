// Test that calling a builtin library function (e.g. __builtin_hypotf) correctly
// looks up the real standard library declaration from an imported module or header
// and attaches attributes such as dllimport, even if the builtin is implicitly
// declared noexcept while the C standard library declaration lacks noexcept.
// Also test that calling the builtin without prior declaration in the AST
// continues to work, falling back to an external symbol without dllimport.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// 1. Clang Modules enabled: calling __builtin_hypotf attaches dllimport from imported math.h
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -I%t -emit-llvm %t/main.cpp -o - | FileCheck %s --check-prefix=CHECK-DLLIMPORT

// 2. Non-modules (textual include): calling __builtin_hypotf attaches dllimport from math.h
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -I%t -emit-llvm %t/main.cpp -o - | FileCheck %s --check-prefix=CHECK-DLLIMPORT

// 3. No math.h included: calling __builtin_hypotf directly works and emits external symbol without dllimport
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fdeclspec -emit-llvm %t/direct_builtin.cpp -o - | FileCheck %s --check-prefix=CHECK-NO-DLLIMPORT

// CHECK-DLLIMPORT: declare dllimport float @hypotf(float noundef, float noundef)

// CHECK-NO-DLLIMPORT: declare dso_local float @hypotf(float noundef, float noundef)

//--- math.h
extern "C" __declspec(dllimport) float hypotf(float x, float y);

//--- module.modulemap
module Math {
  header "math.h"
}

//--- main.cpp
#include "math.h"

extern "C" float test_func(float x, float y) {
  return __builtin_hypotf(x, y);
}

//--- direct_builtin.cpp
extern "C" float test_func(float x, float y) {
  return __builtin_hypotf(x, y);
}
