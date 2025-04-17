// RUN: %clang_cc1 -triple aarch64-unknown-none-gnu %s -target-feature +reserve-x4 -target-feature +reserve-x15 -verify -verify=no_x18 -fsyntax-only
// RUN: %clang_cc1 -triple aarch64-unknown-android %s -target-feature +reserve-x4 -target-feature +reserve-x15 -verify -fsyntax-only

register int w0 __asm__ ("w0");
register long x0 __asm__ ("x0");
register char i1 __asm__ ("x15"); // expected-error {{size of register 'x15' does not match variable size}}
register long long l2 __asm__ ("w15"); // expected-error {{size of register 'w15' does not match variable size}}
register int w3 __asm__ ("w3"); // expected-error {{register 'w3' unsuitable for global register variables on this target}}
register long x3 __asm__ ("x3"); // expected-error {{register 'x3' unsuitable for global register variables on this target}}
register int w4 __asm__ ("w4");
register long x4 __asm__ ("x4");
register int w18 __asm__ ("w18"); // no_x18-error {{register 'w18' unsuitable for global register variables on this target}}
register long x18 __asm__ ("x18"); // no_x18-error {{register 'x18' unsuitable for global register variables on this target}}
