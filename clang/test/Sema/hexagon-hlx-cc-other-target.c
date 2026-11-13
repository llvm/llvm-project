// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -verify %s

// hlxcc and preserve_hlx are Hexagon-only attributes.
void __attribute__((hlxcc)) f(void);        // expected-warning {{unknown attribute 'hlxcc' ignored}}
void __attribute__((preserve_hlx)) g(void); // expected-warning {{unknown attribute 'preserve_hlx' ignored}}
