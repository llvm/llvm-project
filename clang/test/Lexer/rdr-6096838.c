/* RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -verify %s
 * RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=gnu89 -fsyntax-only -verify %s
 rdar://6096838
 */
// expected-no-diagnostics

long double d = 0x0.0000003ffffffff00000p-16357L;
