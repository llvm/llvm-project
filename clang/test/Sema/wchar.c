// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -fsyntax-only -fwchar-type=short -fno-signed-wchar -verify -DSHORT_WCHAR

typedef __WCHAR_TYPE__ wchar_t;

#if defined(_WIN32) || defined(_M_IX86) || defined(__CYGWIN__) \
 || defined(_M_X64) || defined(__ORBIS__) || defined(__PROSPERO__) \
 || defined(SHORT_WCHAR) || (defined(_AIX) && !defined(__64BIT__))
  #define WCHAR_T_TYPE unsigned short
#elif defined(__aarch64__)
  // See AArch64TargetInfo constructor -- unsigned on non-darwin non-OpenBSD non-NetBSD.
  #if defined(__OpenBSD__) || defined(__APPLE__) || defined(__NetBSD__)
    #define WCHAR_T_TYPE int
  #else
    #define WCHAR_T_TYPE unsigned int
  #endif
#elif defined(__arm) || defined(__MVS__) || (defined(_AIX) && defined(__64BIT__))
  #define WCHAR_T_TYPE unsigned int
#elif defined(__sun)
  #if defined(__LP64__)
    #define WCHAR_T_TYPE int
  #else
    #define WCHAR_T_TYPE long
  #endif
#else /* Solaris, Linux, non-arm64 macOS, ... */
  #define WCHAR_T_TYPE int
#endif
 
int check_wchar_size[sizeof(*L"") == sizeof(wchar_t) ? 1 : -1];
 
void foo(void) {
  WCHAR_T_TYPE t1[] = L"x";
  wchar_t tab[] = L"x";
  WCHAR_T_TYPE t2[] = "x";     // expected-error {{initializing wide char array with non-wide string literal}}
  char t3[] = L"x";   // expected-error {{initializing char array with wide string literal}}
}
