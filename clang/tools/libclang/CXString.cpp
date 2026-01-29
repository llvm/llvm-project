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
#include "clang-c/CXString.h"
#include "clang-c/Index.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TrailingObjects.h"

using namespace clang;

namespace {
/// Describes the kind of underlying data in CXString.
enum CXStringFlag : unsigned {
  /// CXString contains a 'const char *' that it doesn't own.
  CXS_Unmanaged = 0,

  /// CXString contains a 'const char *' that is allocated with malloc().
  /// WARNING: do not use this variant outside c-index-test!
  CXS_Malloc,

  /// CXString contains a 'CStringImpl' that it allocated with malloc().
  CXS_MallocWithSize,

  /// CXString contains a CXStringBuf that needs to be returned to the
  /// CXStringPool.
  CXS_StringBuf
};

struct CStringImpl final : llvm::TrailingObjects<CStringImpl, char> {
  size_t length;

  CStringImpl(size_t len) : length(len) {}

  char *Buffer() { return getTrailingObjects(); }
  const char *Buffer() const { return getTrailingObjects(); }

  static CStringImpl *Create(size_t length) {
    void *Mem = llvm::safe_malloc(totalSizeToAlloc<char>(length + 1));
    return new (Mem) CStringImpl(length);
  }
};
} // end namespace

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
  auto *ptr = CStringImpl::Create(String.size());
  auto *buf = ptr->Buffer();
  memcpy(buf, String.data(), String.size());
  buf[String.size()] = 0;

  CXString Result;
  Result.data = ptr;
  Result.private_flags = static_cast<unsigned>(CXS_MallocWithSize);
  return Result;
}

CXString createCXString(CXStringBuf *buf) {
  CXString Str;
  Str.data = buf;
  Str.private_flags = static_cast<unsigned>(CXS_StringBuf);
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
  return static_cast<CXStringFlag>(str.private_flags) == CXS_StringBuf;
}

} // end namespace cxstring
} // end namespace clang

//===----------------------------------------------------------------------===//
// libClang public APIs.
//===----------------------------------------------------------------------===//

const char *clang_getCString(CXString string) {
  return clang_getCStringInfo(string).string;
}

CStringInfo clang_getCStringInfo(CXString string) {
  switch (static_cast<CXStringFlag>(string.private_flags)) {
  case CXS_Unmanaged:
  case CXS_Malloc: {
    auto *ptr = static_cast<const char *>(string.data);
    return {ptr, strlen(ptr)};
  }
  case CXS_MallocWithSize: {
    auto *ptr = static_cast<const CStringImpl *>(string.data);
    return {ptr->Buffer(), ptr->length};
  }
  case CXS_StringBuf: {
    auto *ptr = static_cast<const cxstring::CXStringBuf *>(string.data);
    return {ptr->Data.data(), ptr->Data.size()};
  }
  }
  llvm_unreachable("Invalid CXString::private_flags");
}

void clang_disposeString(CXString string) {
  switch (static_cast<CXStringFlag>(string.private_flags)) {
  case CXS_Unmanaged:
    return;
  case CXS_Malloc:
  case CXS_MallocWithSize:
    if (string.data) {
      // Safety:
      // - the malloc'ed string can be free'ed
      // - CStringImpl was malloc'ed and has trivial destructor
      free(const_cast<void *>(string.data));
    }
    return;
  case CXS_StringBuf:
    static_cast<cxstring::CXStringBuf *>(const_cast<void *>(string.data))
        ->dispose();
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

