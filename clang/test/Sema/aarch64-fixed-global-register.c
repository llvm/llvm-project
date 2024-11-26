// RUN: %clang_cc1 -triple aarch64-unknown-none-gnu %s -target-feature +reserve-x4 -target-feature +reserve-x15 -verify -verify=no_x18 -fsyntax-only
// RUN: %clang_cc1 -triple aarch64-unknown-android %s -target-feature +reserve-x4 -target-feature +reserve-x15 -verify -fsyntax-only

register long x0 __asm__ ("x0");
register char i1 __asm__ ("x15"); // expected-error {{size of register 'x15' does not match variable size}}
register long long l2 __asm__ ("w14"); // expected-error {{size of register 'w14' does not match variable size}}
register long x3 __asm__ ("x3"); // expected-error {{register 'x3' unsuitable for global register variables on this target}}
register long x4 __asm__ ("x4");
register long x18 __asm__ ("x18"); // no_x18-error {{register 'x18' unsuitable for global register variables on this target}}
