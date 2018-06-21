// RUN: %clang_cc1 -emit-llvm %s -o - -triple i386-pc-win32 | FileCheck %s --check-prefix=CHECK-WIN
// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-apple-darwin | FileCheck %s --check-prefix=CHECK-DAR
// This should pass for any endianness combination of host and target.

// This bit is taken from Sema/wchar.c so we can avoid the wchar.h include.
typedef __WCHAR_TYPE__ wchar_t;
#if defined(_WIN32) || defined(_M_IX86) || defined(__CYGWIN__) \
  || defined(_M_X64) || defined(SHORT_WCHAR)
  #define WCHAR_T_TYPE unsigned short
#elif defined(__sun)
  #define WCHAR_T_TYPE long
#else /* Solaris. */
  #define WCHAR_T_TYPE int
#endif


// CHECK-DAR: private unnamed_addr constant [18 x i32] [i32 84,
// CHECK-WIN: linkonce_odr unnamed_addr constant [18 x i16] [i16 84,
extern void foo(const wchar_t* p);
int main (int argc, const char * argv[])
{
 foo(L"This is some text");
 return 0;
}
