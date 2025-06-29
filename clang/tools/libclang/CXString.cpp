//===- CXString.cpp - Routines for manipulating CXStrings -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  /// CXString contains a 'CStringImpl' that it allocated with malloc().
  CXS_MallocWithSize,

  /// CXString contains a CXStringBuf that needs to be returned to the
  /// CXStringPool.
  CXS_StringBuf
};

struct CStringImpl {
  size_t length;
  char buffer[sizeof(length)];
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

  return createDup(StringRef(String));
}

CXString createRef(StringRef String) {
  if (!String.data())
    return createNull();

  // If the string is empty, it might point to a position in another string
  // while having zero length. Make sure we don't create a reference to the
  // larger string.
  if (String.empty())
    return createEmpty();

  return createDup(String);
}

CXString createDup(StringRef String) {
  auto toAllocate =
      sizeof(size_t) + std::max(sizeof(size_t), String.size() + 1);
  assert(toAllocate >= sizeof(CStringImpl));
  auto ptr = static_cast<CStringImpl *>(llvm::safe_malloc(toAllocate));

  ptr->length = String.size();
  memcpy(ptr->buffer, String.data(), String.size());
  ptr->buffer[String.size()] = 0;

  CXString Result;
  Result.data = ptr;
  Result.private_flags = (unsigned)CXS_MallocWithSize;
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

CStringInfo clang_getCStringInfo(CXString string) {
  switch ((CXStringFlag)string.private_flags) {
  case CXS_Unmanaged: {
    auto ptr = static_cast<const char *>(string.data);
    return {ptr, strlen(ptr)};
  }
  case CXS_MallocWithSize: {
    auto ptr = static_cast<const CStringImpl *>(string.data);
    return {ptr->buffer, ptr->length};
  }
  case CXS_StringBuf: {
    auto ptr = static_cast<const cxstring::CXStringBuf *>(string.data);
    return {ptr->Data.data(), ptr->Data.size()};
  }
  }
  llvm_unreachable("Invalid CXString::private_flags");
}

void clang_disposeString(CXString string) {
  switch ((CXStringFlag) string.private_flags) {
    case CXS_Unmanaged:
      return;
    case CXS_MallocWithSize:
      if (string.data)
        free(const_cast<void *>(string.data));
      return;
    case CXS_StringBuf:
      static_cast<cxstring::CXStringBuf *>(
          const_cast<void *>(string.data))->dispose();
      return;
  }
  llvm_unreachable("Invalid CXString::private_flags");
}

void clang_disposeStringSet(CXStringSet *set) {
  for (unsigned SI = 0, SE = set->Count; SI < SE; ++SI)
    clang_disposeString(set->Strings[SI]);
  delete[] set->Strings;
  delete set;
}

