/* RUN: %clang_cc1 -triple x86_64-unknown-unknown -pedantic -std=gnu89 -fsyntax-only -verify %s
 */

long double d = 0x0.0000003ffffffff00000p-16357L; /* expected-warning {{hexadecimal floating constants are a C99 feature}} */
