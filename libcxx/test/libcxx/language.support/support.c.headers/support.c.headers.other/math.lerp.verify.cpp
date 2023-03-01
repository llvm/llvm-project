//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When built with modules, this test gives diagnostics like
//  declaration of 'lerp' must be imported from module 'std.compat.cmath'
//  before it is required
// therefore disable the test in this configuration.
// UNSUPPORTED: modules-build

// <math.h>

// [support.c.headers.other]/1
//   ... except for the functions described in [sf.cmath], the
//   std::lerp function overloads ([c.math.lerp]) ...

#include <math.h>

void f() {
  {
    float f;
    ::lerp(f, f, f);    // expected-error {{no member named 'lerp' in the global namespace}}
    std::lerp(f, f, f); // expected-error {{no member named 'lerp' in namespace 'std'}}
  }
  {
    double d;
    ::lerp(d, d, d);    // expected-error {{no member named 'lerp' in the global namespace}}
    std::lerp(d, d, d); // expected-error {{no member named 'lerp' in namespace 'std'}}
  }
  {
    long double l;
    ::lerp(l, l, l);    // expected-error {{no member named 'lerp' in the global namespace}}
    std::lerp(l, l, l); // expected-error {{no member named 'lerp' in namespace 'std'}}
  }
}
