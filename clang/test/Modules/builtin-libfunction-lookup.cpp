// Test that calling built-in library functions like __builtin_hypotf under Clang Modules
// triggers AST name lookup for the target C function (hypotf). This ensures that
// module-defined inline wrappers (e.g. MSVC UCRT's hypotf wrapper calling _hypotf)
// are lazily deserialized from PCMs rather than emitting external non-existent function calls.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -xc++ -emit-module -fmodules -fmodule-name=ucrt -fmodule-map-file=ucrt.modulemap -triple x86_64-pc-windows-msvc -fms-extensions -I. ucrt.modulemap -o ucrt.pcm
// RUN: %clang_cc1 -xc++ -emit-module -fmodules -fmodule-name=std -fmodule-map-file=std.modulemap -fmodule-file=ucrt=ucrt.pcm -triple x86_64-pc-windows-msvc -fms-extensions -I. std.modulemap -o std.pcm
// RUN: %clang_cc1 -xc++ -emit-llvm -fmodules -fmodule-map-file=std.modulemap -fmodule-map-file=ucrt.modulemap -fmodule-file=std=std.pcm -fmodule-file=ucrt=ucrt.pcm -triple x86_64-pc-windows-msvc -fms-extensions -I. main.cc -o - | FileCheck %s

//--- corecrt_math.h
#ifndef MOCK_CORECRT_MATH_H
#define MOCK_CORECRT_MATH_H
extern "C" {
__declspec(dllimport) float __cdecl _hypotf(float x, float y);
inline float __cdecl hypotf(float x, float y) {
  return _hypotf(x, y);
}
}
#endif

//--- math.h
#ifndef MOCK_MATH_H
#define MOCK_MATH_H
#include "corecrt_math.h"
#endif

//--- __math/hypot.h
#ifndef MOCK_MATH_HYPOT_H
#define MOCK_MATH_HYPOT_H
inline float hypot(float x, float y) {
  return __builtin_hypotf(x, y);
}
#endif

//--- cmath
#ifndef MOCK_CMATH
#define MOCK_CMATH
#include "math.h"
#include "__math/hypot.h"
#endif

//--- std.modulemap
module std {
  module cmath {
    header "cmath"
    export *
  }
  module math_hypot {
    header "__math/hypot.h"
    export *
  }
}

//--- ucrt.modulemap
module ucrt {
  module math {
    header "math.h"
    export *
  }
  module corecrt_math {
    header "corecrt_math.h"
    export *
  }
}

//--- main.cc
#include "cmath"

float test_call(float x, float y) {
  return hypot(x, y);
}

// CHECK: define linkonce_odr dso_local float @hypotf(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK: call float @_hypotf(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK: declare dllimport float @_hypotf(float noundef, float noundef)
