// RUN: %clang_cc1 -fsyntax-only -verify=noncxx,expected,no-nonstring -Wunterminated-string-initialization %s
// RUN: %clang_cc1 -fsyntax-only -verify=noncxx,expected,no-nonstring -Wextra %s
// RUN: %clang_cc1 -fsyntax-only -verify=noncxx,expected,compat -Wc++-unterminated-string-initialization %s
// RUN: %clang_cc1 -fsyntax-only -verify=noncxx,expected,compat -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,noncxx -Wno-unterminated-string-initialization %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,noncxx -Wc++-compat -Wextra -Wno-unterminated-string-initialization -Wno-c++-unterminated-string-initialization %s

char foo[3] = "foo"; // no-nonstring-warning {{initializer-string for character array is too long, array size is 3 but initializer has size 4 (including the null terminating character); did you mean to use the 'nonstring' attribute?}} \
                        compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
                        cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}
__attribute__((nonstring)) char bar[3] = "bar"; // compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
                                                   cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

struct S {
  char buf[3];
  __attribute__((nonstring)) char fub[3];
} s = { "baz", "bop" }; // no-nonstring-warning {{initializer-string for character array is too long, array size is 3 but initializer has size 4 (including the null terminating character); did you mean to use the 'nonstring' attribute?}} \
                           compat-warning 2 {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
                           cxx-error 2 {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

// Show that we also issue the diagnostic for the other character types as well.
signed char scfoo[3] = "foo";    // no-nonstring-warning {{initializer-string for character array is too long, array size is 3 but initializer has size 4 (including the null terminating character); did you mean to use the 'nonstring' attribute?}} \
                                    compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
                                    cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}
unsigned char ucfoo[3] = "foo";  // no-nonstring-warning {{initializer-string for character array is too long, array size is 3 but initializer has size 4 (including the null terminating character); did you mean to use the 'nonstring' attribute?}} \
                                    compat-warning {{initializer-string for character array is too long for C++, array size is 3 but initializer has size 4 (including the null terminating character)}} \
                                    cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 4 (including the null terminating character)}}

// wchar_t (et al) are a bit funny in that it cannot do string init in C++ at
// all, so they are not tested.

// Show that we properly diagnose incorrect uses of nonstring.
__attribute__((nonstring)) void func(void);       // expected-warning {{'nonstring' attribute only applies to variables and non-static data members}}
__attribute__((nonstring("test"))) char eek1[2];  // expected-error {{'nonstring' attribute takes no arguments}}
__attribute__((nonstring)) int eek2;              // expected-warning {{'nonstring' attribute only applies to fields or variables of character array type; type is 'int'}}

// Note, these diagnostics are separate from the "too many initializers"
// diagnostic when you overwrite more than just the null terminator.
char too_big[3] = "this is too big"; // noncxx-warning {{initializer-string for char array is too long}} \
                                        cxx-error {{initializer-string for char array is too long, array size is 3 but initializer has size 16 (including the null terminating character)}}
