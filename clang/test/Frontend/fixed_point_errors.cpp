// RUN: %clang_cc1 -x c++ %s -verify -DWITHOUT_FIXED_POINT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -verify -ffixed-point
// RUN: %clang_cc1 -triple x86_64-w64-windows-gnu -x c++ %s -verify -ffixed-point
// RUN: not %clang_cc1 -triple x86_64-pc-windows-msvc -x c++ %s -ffixed-point 2>&1 | FileCheck %s --check-prefix=MSABI --implicit-check-not=error:
// RUN: not %clang_cc1 -triple x86_64-unknown-uefi -x c++ %s -ffixed-point 2>&1 | FileCheck %s --check-prefix=MSABI --implicit-check-not=error:
// RUN: not %clang --target=x86_64-pc-windows-msvc -x c++ /dev/null -ffixed-point -fsyntax-only 2>&1 | FileCheck %s --check-prefix=MSABI --implicit-check-not=error:
// RUN: not %clang --target=x86_64-unknown-uefi -x c++ /dev/null -ffixed-point -fsyntax-only 2>&1 | FileCheck %s --check-prefix=MSABI --implicit-check-not=error:
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -x c /dev/null -ffixed-point -fsyntax-only
// RUN: %clang_cc1 -triple x86_64-unknown-uefi -x c /dev/null -ffixed-point -fsyntax-only

// MSABI: error: '-ffixed-point' is not supported for the Microsoft C++ ABI

#ifdef WITHOUT_FIXED_POINT
_Accum accum;                           // expected-error{{unknown type name '_Accum'}}
_Fract fract;                           // expected-error{{unknown type name '_Fract'}}
_Sat _Accum sat_accum;                  // expected-error{{unknown type name '_Sat'}}
                                        // expected-error@-1{{expected ';' after top level declarator}}
#endif

int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
#ifdef WITHOUT_FIXED_POINT
float accum_flt = 0.0k;  // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 0.0r;  // expected-error{{invalid suffix 'r' on floating constant}}
#endif

#ifndef WITHOUT_FIXED_POINT
const char *c = 10.0k;  // expected-error{{cannot initialize a variable of type 'const char *' with an rvalue of type '_Accum'}}
#endif
