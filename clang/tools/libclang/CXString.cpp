//===- CXString.cpp - Routines for manipulating CXStrings -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXStrings. It should be the
// only file that has internal knowledge of the encoding of the data in
// CXStrings.
//
//===----------------------------------------------------------------------===//

#include "CXString.h"
#include "CXTranslationUnit.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang-c/Index.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::cxstring;

enum CXStringFlag { CXS_Unmanaged, CXS_Malloc, CXS_StringBuf };

//===----------------------------------------------------------------------===//
// Basic generation of CXStrings.
//===----------------------------------------------------------------------===//

CXString cxstring::createCXString(const char *String, bool DupString){
  CXString Str;
  if (DupString) {
    Str.data = strdup(String);
    Str.private_flags = (unsigned) CXS_Malloc;
  } else {
    Str.data = (void*)String;
    Str.private_flags = (unsigned) CXS_Unmanaged;
  }
  return Str;
}

CXString cxstring::createCXString(StringRef String, bool DupString) {
  CXString Result;
  if (DupString || (!String.empty() && String.data()[String.size()] != 0)) {
    char *Spelling = (char *)malloc(String.size() + 1);
    memmove(Spelling, String.data(), String.size());
    Spelling[String.size()] = 0;
    Result.data = Spelling;
    Result.private_flags = (unsigned) CXS_Malloc;
  } else {
    Result.data = (void*) String.data();
    Result.private_flags = (unsigned) CXS_Unmanaged;
  }
  return Result;
}

CXString cxstring::createCXString(CXStringBuf *buf) {
  CXString Str;
  Str.data = buf;
  Str.private_flags = (unsigned) CXS_StringBuf;
  return Str;
}


//===----------------------------------------------------------------------===//
// String pools.
//===----------------------------------------------------------------------===//

  
typedef std::vector<CXStringBuf *> CXStringPool;

void *cxstring::createCXStringPool() {
  return new CXStringPool();
}

void cxstring::disposeCXStringPool(void *p) {
  CXStringPool *pool = static_cast<CXStringPool*>(p);
  if (pool) {
    for (CXStringPool::iterator I = pool->begin(), E = pool->end();
         I != E; ++I) {
      delete *I;
    }
    delete pool;
  }
}

CXStringBuf *cxstring::getCXStringBuf(CXTranslationUnit TU) {
  CXStringPool *pool = static_cast<CXStringPool*>(TU->StringPool);
  if (pool->empty())
    return new CXStringBuf(TU);
  CXStringBuf *buf = pool->back();
  buf->Data.clear();
  pool->pop_back();
  return buf;
}

void cxstring::disposeCXStringBuf(CXStringBuf *buf) {
  if (buf)
    static_cast<CXStringPool*>(buf->TU->StringPool)->push_back(buf);
}

bool cxstring::isManagedByPool(CXString str) {
  return ((CXStringFlag) str.private_flags) == CXS_StringBuf;
}

//===----------------------------------------------------------------------===//
// libClang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {
const char *clang_getCString(CXString string) {
  if (string.private_flags == (unsigned) CXS_StringBuf) {
    return ((CXStringBuf*)string.data)->Data.data();
  }
  return (const char*) string.data;
}

void clang_disposeString(CXString string) {
  switch ((CXStringFlag) string.private_flags) {
    case CXS_Unmanaged:
      break;
    case CXS_Malloc:
      if (string.data)
        free((void*)string.data);
      break;
    case CXS_StringBuf:
      disposeCXStringBuf((CXStringBuf *) string.data);
      break;
  }
}
} // end: extern "C"

