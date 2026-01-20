// RUN: %clang_cc1 -verify %s -pedantic-errors
// RUN: %clang_cc1 -verify %s -pedantic-errors -DINLINE
// RUN: %clang_cc1 -verify %s -pedantic-errors -DSTATIC
// RUN: %clang_cc1 -verify %s -pedantic-errors -std=c++11 -DCONSTEXPR
// RUN: %clang_cc1 -verify %s -std=c++11 -DDELETED

#if INLINE
inline // expected-error {{'main' is not allowed to be declared inline}}
#elif STATIC
static // expected-error {{'main' is not allowed to be declared static}}
#elif CONSTEXPR
constexpr // expected-error {{'main' is not allowed to be declared constexpr}}
#endif
int main(int argc, char **argv)
#if DELETED
  = delete; // expected-error {{'main' is not allowed to be deleted}}
#else
{
  int (*pmain)(int, char**) = &main; // expected-error {{referring to 'main' within an expression is a Clang extension}}

  if (argc)
    main(0, 0); // expected-error {{referring to 'main' within an expression is a Clang extension}}
}
#endif
