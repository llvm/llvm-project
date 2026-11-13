// FIXME: Tracked by Github Issue 241514
// RUN: not --crash %clang_cc1 -triple s390x-none-zos -fexec-charset IBM-1047 %s -std=c++23 -fsyntax-only -verify
// REQUIRES: asserts

// In IBM-1047, \x68 is U+00C7 (LATIN CAPITAL LETTER C WITH CEDILLA), so the
// asm string below is converted to UTF-8 with a different length than the
// source token. Computing diagnostic locations within the string must not
// assert.
void f(unsigned x) {
  __asm__("\tXR %[\x68\x68\x68],%[z]" : [\u{c7}\u{c7}\u{c7}] "=r"(x)); // expected-error {{unknown symbolic operand name in inline assembly string}}
}
