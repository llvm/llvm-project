// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsyntax-only -fshort-wchar -verify -DSHORT_WCHAR

typedef __WCHAR_TYPE__ wchar_t;

#if defined(_WIN32) || defined(_M_IX86) || defined(__CYGWIN__) \
 || defined(_M_X64) || defined(__ORBIS__) || defined(SHORT_WCHAR)
  #define WCHAR_T_TYPE unsigned short
#elif defined(__arm) || defined(__aarch64__)
  #define WCHAR_T_TYPE unsigned int
#elif defined(__sun)
  #define WCHAR_T_TYPE long
#else /* Solaris. */
  #define WCHAR_T_TYPE int
#endif
 
int check_wchar_size[sizeof(*L"") == sizeof(wchar_t) ? 1 : -1];
 
void foo() {
  WCHAR_T_TYPE t1[] = L"x";
  wchar_t tab[] = L"x";
  WCHAR_T_TYPE t2[] = "x";     // expected-error {{initializing wide char array with non-wide string literal}}
  char t3[] = L"x";   // expected-error {{initializing char array with wide string literal}}
}
