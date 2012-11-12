// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

// This bit is taken from Sema/wchar.c so we can avoid the wchar.h include.
typedef __WCHAR_TYPE__ wchar_t;
#if defined(_WIN32) || defined(_M_IX86) || defined(__CYGWIN__) \
  || defined(_M_X64) || defined(SHORT_WCHAR)
  #define WCHAR_T_TYPE unsigned short
#elif defined(__sun) || defined(__AuroraUX__)
  #define WCHAR_T_TYPE long
#else /* Solaris or AuroraUX. */
  #define WCHAR_T_TYPE int
#endif

signed short _iodbcdm_sqlerror( )
{
  wchar_t _sqlState[6] = { L"\0" };
}
