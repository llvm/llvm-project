/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify -pedantic -Wno-strict-prototypes %s
 */

int main() {

} /* expected-warning {{non-void 'main' function does not return a value}} */
