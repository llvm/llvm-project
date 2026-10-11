// RUN: %clang_cc1 -triple s390x-none-zos -fexec-charset IBM-1047 %s -std=c++23 -fsyntax-only -verify
// RUN: %clang_cc1 -triple s390x-none-zos -fexec-charset UTF-8 %s -std=c++23 -fsyntax-only -verify

long inline_asm_ucn_named_operand(long a, long b) {
  // The \N{...} escape encodes to U+0130.  IBM-1047 has no code point for
  // U+0130, so the conversion reports an error.
  asm("\tLGR %0, %[\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}]\n" // expected-error-re {{encoding conversion failed: {{.*}}}}
      : "=r"(a)
      : [\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}] "r"(b));
  return a;
}
