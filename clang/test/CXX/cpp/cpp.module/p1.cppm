// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -triple x86_64-linux-gnu -DTEST=1 -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -triple x86_64-linux-gnu -DTEST=2 -verify

module;
export module x;
#include "version.h"
#if TEST == 1
export module VERSION;  // expected-error {{module declaration cannot be a macro}}
#endif // TEST == 1

#if TEST == 2
export module A.B;      // expected-error {{module declaration cannot be a macro}}
#endif // TEST == 2
