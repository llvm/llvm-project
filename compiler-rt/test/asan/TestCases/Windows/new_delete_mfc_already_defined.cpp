// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t

// test fixing new/delete already defined
// as long as it finishes linking, it should be good

// before fix:
// nafxcw.lib(afxmem.obj) : warning LNK4006: "void * __cdecl operator new(unsigned int)" (??2@YAPAXI@Z) already defined in clang_rt.asan_cxx-i386.lib(asan_win_new_scalar_thunk.cpp.obj); second definition ignored
// nafxcw.lib(afxmem.obj) : warning LNK4006: "void __cdecl operator delete(void *)" (??3@YAXPAX@Z) already defined in clang_rt.asan_cxx-i386.lib(asan_win_delete_scalar_thunk.cpp.obj); second definition ignored

#ifdef _DLL
#  define _AFXDLL
#endif

#include "afxglobals.h"

int AFX_CDECL AfxCriticalNewHandler(size_t nSize);

int main(int argc, char **argv) {
  AFX_MODULE_THREAD_STATE *pState = AfxGetModuleThreadState();
  _PNH pnhOldHandler = AfxSetNewHandler(&AfxCriticalNewHandler);
  AfxSetNewHandler(pnhOldHandler);
  puts("Pass");
  return 0;
}
