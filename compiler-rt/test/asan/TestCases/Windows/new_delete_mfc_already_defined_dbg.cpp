// RUN: %clang_cl_asan -Od %s -Fe%t /MT /link /WX
// RUN: %env_asan_opts=alloc_dealloc_mismatch=true %run %t

// test fixing new already defined and mismatch between allocation and deallocation APis

// before fix:
// nafxcwd.lib(afx_new_scalar.obj) : error LNK2005: "void * __cdecl operator new(unsigned int)" (??2@YAPAXI@Z) already defined in clang_rt.asan_cxx_dbg-i386.lib(asan_win_new_scalar_thunk.cpp.obj)
// Address Sanitizer error: mismatch between allocation and deallocation APis

#ifdef _DLL
#  define _AFXDLL
#endif

#include <SDKDDKVer.h>
#include <afxglobals.h>

int main() {
  int *normal = new int;
  int *debug = DEBUG_NEW int;

  delete normal;
  delete debug;

  return 0;
}
