// RUN: %clang_cc1 %s -fsyntax-only -std=c99 -verify
// RUN: %clang_cc1 %s -fsyntax-only -std=c11 -Wc99-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -std=c++03 -Wc++11-compat -verify
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -std=c++11 -Wc++98-compat -verify

// Identifier characters
extern char a\u01F6; // C11, C++11
extern char a\u00AA; // C99, C11, C++11
extern char a\u0384; // C++03, C11, C++11
extern char a\u0E50; // C99, C++03, C11, C++11
extern char a\uFFFF; // none





// Identifier initial characters
extern char \u0E50; // C++03, C11, C++11
extern char \u0300; // disallowed in C99/C++03
extern char \u0D61; // C99, C11, C++03, C++11







// Disallowed everywhere
#define A \u0000 // expected-error{{control character}}
#define B \u001F // expected-error{{control character}}
#define C \u007F // expected-error{{control character}}
#define D \u009F // expected-error{{control character}}
#define E \uD800 // C++03 allows UCNs representing surrogate characters!






#if __cplusplus
// expected-error@9 {{character <U+0384> not allowed in an identifier}}
// expected-error@11 {{character <U+FFFF> not allowed in an identifier}}
// expected-error@18 {{expected unqualified-id}}
# if __cplusplus >= 201103L
// C++11
// expected-error@19 {{expected unqualified-id}}
// expected-error@33 {{invalid universal character}}

# else
// C++03
// expected-error@19 {{expected unqualified-id}}
// expected-warning@33 {{universal character name refers to a surrogate character}}

# endif
#else
# if __STDC_VERSION__ >= 201112L
// C11
// expected-warning@7 {{using this character in an identifier is incompatible with C99}}
// expected-warning@9 {{using this character in an identifier is incompatible with C99}}
// expected-error@11 {{character <U+FFFF> not allowed in an identifier}}
// expected-warning@18 {{starting an identifier with this character is incompatible with C99}}
// expected-error@19 {{expected identifier}}
// expected-error@33 {{invalid universal character}}

# else
// C99
// expected-error@7 {{not allowed in an identifier}}
// expected-error@9 {{not allowed in an identifier}}
// expected-error@11 {{not allowed in an identifier}}
// expected-error@18 {{expected identifier}}
// expected-error@19 {{expected identifier}}
// expected-error@33 {{invalid universal character}}

# endif
#endif
