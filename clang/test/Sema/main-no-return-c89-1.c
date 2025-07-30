/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -Wno-strict-prototypes -pedantic %s
 * RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -Wno-strict-prototypes -Wmain-return-type %s
 * RUN: %clang_cc1 -std=c89 -fsyntax-only -verify=implicit-main-return -Wno-strict-prototypes -pedantic -Wno-main-return-type %s
 */

/* implicit-main-return-no-diagnostics */

int main() {

} /* expected-warning {{implicit '0' return value from 'main' is a C99 extension}} */
