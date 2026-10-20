// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -ffreestanding %s
// RUN: %clang_cc1 -std=c99 -verify=expected,ped -Wall -pedantic -ffreestanding %s

/* WG14 N3346: Yes
 * Slay Some Earthly Demons VIII
 *
 * Updates some undefined behavior during initialization to instead be a
 * constraint violation.
 */

// The initializer for a scalar shall be a single expression, optionally
// enclosed in braces, or it shall be an empty initializer.
int i = 12, j = {12}, k = {}; // ped-warning {{use of an empty initializer is a C23 extension}}

struct S {
  int i;
  float f;
  int : 0;
  char c;
};

void test1(void) {
  // The initializer for an object that has structure or union type shall be
  // either a single expression that has compatible type or a brace-enclosed
  // list of initializers for the elements or named members.
  struct S s1 = { 1, 1.2f, 'a' };
  struct S s2 = s1;

  // Despite being structurally identical to S, T is not compatible with S.
  struct T { int i; float f; int : 0; char c; } t;
  struct S s3 = t; // expected-error {{initializing 'struct S' with an expression of incompatible type 'struct T'}}
}

void test2(void) {
  typedef __WCHAR_TYPE__ wchar_t;

  // The initializer for an array shall be either a string literal, optionally
  // enclosed in braces, or a brace-enclosed list of initializers for the
  // elements. An array initialized by character string literal or UTF-8 string
  // literal shall have a character type as element type. An array initialized
  // with a wide string literal shall have element type compatible with a
  // qualified or unqualified wchar_t, char16_t, or char32_t, and the string
  // literal shall have the corresponding encoding prefix (L, u, or U,
  // respectively).
  char str1[] = "string literal";
  char str2[] = { "string literal" };

  float str5[] = "this doesn't work";          // expected-error {{array initializer must be an initializer list}}
  float str6[] = { "this also doesn't work" }; // expected-error {{initializing 'float' with an expression of incompatible type 'char[23]'}}

  wchar_t str7[] = L"string literal";
  wchar_t str8[] = { L"string literal" };

#if __STDC_VERSION__ >= 201112L
  typedef __CHAR16_TYPE__ char16_t;
  typedef __CHAR32_TYPE__ char32_t;

  char str3[] = u8"string literal";
  char str4[] = { u8"string literal" };

  char16_t str9[] = u"string literal";
  char16_t str10[] = { u"string literal" };
  char32_t str11[] = U"string literal";
  char32_t str12[] = { U"string literal" };

  char16_t str15[] = "nope";     // expected-error {{initializing wide char array with non-wide string literal}}
  char16_t str16[] = { "nope" }; // expected-error-re {{incompatible pointer to integer conversion initializing 'char16_t' (aka '{{.*}}') with an expression of type 'char[5]'}}
  char32_t str17[] = "nope";     // expected-error {{initializing wide char array with non-wide string literal}}
  char32_t str18[] = { "nope" }; // expected-error-re {{incompatible pointer to integer conversion initializing 'char32_t' (aka '{{.*}}') with an expression of type 'char[5]'}}
#endif

  wchar_t str13[] = "nope";      // expected-error {{initializing wide char array with non-wide string literal}}
  wchar_t str14[] = { "nope" };  // expected-error-re {{incompatible pointer to integer conversion initializing 'wchar_t' (aka '{{.*}}') with an expression of type 'char[5]'}}
}
