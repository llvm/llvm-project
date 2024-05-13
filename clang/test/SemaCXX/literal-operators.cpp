// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s

#include <stddef.h>

struct tag {
  void operator "" _tag_bad (const char *); // expected-error {{literal operator 'operator""_tag_bad' must be in a namespace or global scope}}
  friend void operator "" _tag_good (const char *);
};

namespace ns { void operator "" _ns_good (const char *); }

// Check extern "C++" declarations
extern "C++" void operator "" _extern_good (const char *);
extern "C++" { void operator "" _extern_good (const char *); }

void fn () { void operator "" _fn_good (const char *); }

// One-param declarations (const char * was already checked)
void operator "" _good (char);
void operator "" _good (wchar_t);
void operator "" _good (char16_t);
void operator "" _good (char32_t);
void operator "" _good (unsigned long long);
void operator "" _good (long double);

// Two-param declarations
void operator "" _good (const char *, size_t);
void operator "" _good (const wchar_t *, size_t);
void operator "" _good (const char16_t *, size_t);
void operator "" _good (const char32_t *, size_t);

// Check typedef and array equivalences
void operator "" _good (const char[]);
typedef const char c;
void operator "" _good (c*);

// Check extra cv-qualifiers
void operator "" _cv_good (volatile const char *, const size_t); // expected-error {{invalid literal operator parameter type 'const volatile char *', did you mean 'const char *'?}}

// Template declaration
template <char...> void operator "" _good ();

template <typename...> void operator "" _invalid(); // expected-error {{template parameter list for literal operator must be either 'char...' or 'typename T, T...'}}
template <wchar_t...> void operator "" _invalid();  // expected-error {{template parameter list for literal operator must be either 'char...' or 'typename T, T...'}}
template <unsigned long long...> void operator "" _invalid();  // expected-error {{template parameter list for literal operator must be either 'char...' or 'typename T, T...'}}

_Complex float operator""if(long double); // expected-warning {{reserved}}
_Complex float test_if_1() { return 2.0f + 1.5if; };
void test_if_2() { "foo"if; } // expected-error {{no matching literal operator for call to 'operator""if'}}

template<typename T> void dependent_member_template() {
  T().template operator""_foo<int>(); // expected-error {{'operator""_foo' following the 'template' keyword cannot refer to a dependent template}}
}

namespace PR51142 {
// This code previously crashed due to a null template parameter declaration.
template<typename T> // expected-error {{template parameter list for literal operator must be either 'char...' or 'typename T, T...'}}
constexpr auto operator ""_l();
}
