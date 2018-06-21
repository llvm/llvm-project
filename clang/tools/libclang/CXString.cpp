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
#include "clang-c/Index.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

/// Describes the kind of underlying data in CXString.
enum CXStringFlag {
  /// CXString contains a 'const char *' that it doesn't own.
  CXS_Unmanaged,

  /// CXString contains a 'const char *' that it allocated with malloc().
  CXS_Malloc,

  /// CXString contains a CXStringBuf that needs to be returned to the
  /// CXStringPool.
  CXS_StringBuf
};

namespace clang {
namespace cxstring {

//===----------------------------------------------------------------------===//
// Basic generation of CXStrings.
//===----------------------------------------------------------------------===//

CXString createEmpty() {
  CXString Str;
  Str.data = "";
  Str.private_flags = CXS_Unmanaged;
  return Str;
}

CXString createNull() {
  CXString Str;
  Str.data = nullptr;
  Str.private_flags = CXS_Unmanaged;
  return Str;
}

CXString createRef(const char *String) {
  if (String && String[0] == '\0')
    return createEmpty();

  CXString Str;
  Str.data = String;
  Str.private_flags = CXS_Unmanaged;
  return Str;
}

CXString createDup(const char *String) {
  if (!String)
    return createNull();

  if (String[0] == '\0')
    return createEmpty();

  CXString Str;
  Str.data = strdup(String);
  Str.private_flags = CXS_Malloc;
  return Str;
}

CXString createRef(StringRef String) {
  // If the string is not nul-terminated, we have to make a copy.

  // FIXME: This is doing a one past end read, and should be removed! For memory
  // we don't manage, the API string can become unterminated at any time outside
  // our control.

  if (!String.empty() && String.data()[String.size()] != 0)
    return createDup(String);

  CXString Result;
  Result.data = String.data();
  Result.private_flags = (unsigned) CXS_Unmanaged;
  return Result;
}

CXString createDup(StringRef String) {
  CXString Result;
  char *Spelling = static_cast<char *>(malloc(String.size() + 1));
  memmove(Spelling, String.data(), String.size());
  Spelling[String.size()] = 0;
  Result.data = Spelling;
  Result.private_flags = (unsigned) CXS_Malloc;
  return Result;
}

CXString createCXString(CXStringBuf *buf) {
  CXString Str;
  Str.data = buf;
  Str.private_flags = (unsigned) CXS_StringBuf;
  return Str;
}

CXStringSet *createSet(const std::vector<std::string> &Strings) {
  CXStringSet *Set = new CXStringSet;
  Set->Count = Strings.size();
  Set->Strings = new CXString[Set->Count];
  for (unsigned SI = 0, SE = Set->Count; SI < SE; ++SI)
    Set->Strings[SI] = createDup(Strings[SI]);
  return Set;
}


//===----------------------------------------------------------------------===//
// String pools.
//===----------------------------------------------------------------------===//

CXStringPool::~CXStringPool() {
  for (std::vector<CXStringBuf *>::iterator I = Pool.begin(), E = Pool.end();
       I != E; ++I) {
    delete *I;
  }
}

CXStringBuf *CXStringPool::getCXStringBuf(CXTranslationUnit TU) {
  if (Pool.empty())
    return new CXStringBuf(TU);

  CXStringBuf *Buf = Pool.back();
  Buf->Data.clear();
  Pool.pop_back();
  return Buf;
}

CXStringBuf *getCXStringBuf(CXTranslationUnit TU) {
  return TU->StringPool->getCXStringBuf(TU);
}

void CXStringBuf::dispose() {
  TU->StringPool->Pool.push_back(this);
}

bool isManagedByPool(CXString str) {
  return ((CXStringFlag) str.private_flags) == CXS_StringBuf;
}

} // end namespace cxstring
} // end namespace clang

//===----------------------------------------------------------------------===//
// libClang public APIs.
//===----------------------------------------------------------------------===//

const char *clang_getCString(CXString string) {
  if (string.private_flags == (unsigned) CXS_StringBuf) {
    return static_cast<const cxstring::CXStringBuf *>(string.data)->Data.data();
  }
  return static_cast<const char *>(string.data);
}

void clang_disposeString(CXString string) {
  switch ((CXStringFlag) string.private_flags) {
    case CXS_Unmanaged:
      break;
    case CXS_Malloc:
      if (string.data)
        free(const_cast<void *>(string.data));
      break;
    case CXS_StringBuf:
      static_cast<cxstring::CXStringBuf *>(
          const_cast<void *>(string.data))->dispose();
      break;
  }
}

void clang_disposeStringSet(CXStringSet *set) {
  for (unsigned SI = 0, SE = set->Count; SI < SE; ++SI)
    clang_disposeString(set->Strings[SI]);
  delete[] set->Strings;
  delete set;
}

