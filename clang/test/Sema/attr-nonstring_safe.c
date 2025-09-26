// RUN: %clang_cc1 -fsyntax-only -verify=compat,expected -Wunterminated-string-initialization %s -x c
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -Wunterminated-string-initialization %s -x c++
// REQUIRES: !system-linux

#ifndef __cplusplus
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
typedef __WCHAR_TYPE__  wchar_t;
#endif

// C++ is stricter so the following cases should be warned about. In
// C, the following examples are fine.

char foo3[3] = "fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}
char foo1[1] = "\0";   // cxx-error {{initializer-string for char array is too long, array size is 1 but initializer has size 2 (including the null terminating character)}}

struct S {
  char buf[3];
  char fub[3];
} s = { "ba\0", "bo\0" }; // cxx-error 2{{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wc++-compat"
// Test different encodings:
signed char scfoo[3] = "fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			          compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}}
unsigned char ucfoo[3] = "fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}} \
				    compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}}
wchar_t wcfoo[3] = L"fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			       compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			       compat-warning {{identifier 'wchar_t' conflicts with a C++ keyword}}
char16_t c16foo[3] = u"fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			         compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			         compat-warning {{identifier 'char16_t' conflicts with a C++ keyword}}
char32_t c32foo[3] = U"fo\0"; // cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			         compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
			         compat-warning {{identifier 'char32_t' conflicts with a C++ keyword}}
#pragma clang diagnostic pop

// Test list initializer:
signed char scfoo_lst[3] = {'f', 'o', '\0'};
unsigned char ucfoo_lst[3] = {'f', 'o', '\0'};
wchar_t wcfoo_lst[3] = {L'f', L'o', L'\0'};
char16_t c16foo_lst[3] = {u'f', u'o', u'\0'};
char32_t c32foo_lst[3] = {U'f', U'o', U'\0'};

// Declaring an array of size 0 is invalid by C standard but compilers
// may allow it:
char a[0] = ""; // expected-warning {{initializer-string for character array is too long, array size is 0 but initializer has size 1 (including the null terminating character); did you mean to use the 'nonstring' attribute?}} \
		   cxx-error {{initializer-string for char array is too long, array size is 0 but initializer has size 1 (including the null terminating character)}}
char b[1] = ""; // no warn
