//===- Frontend.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/Frontend.h"
#include "clang/AST/Availability.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace llvm::MachO;

namespace clang::installapi {

static StringRef getFileExtension(clang::Language Lang) {
  switch (Lang) {
  default:
    llvm_unreachable("Unexpected language option.");
  case clang::Language::C:
    return ".c";
  case clang::Language::CXX:
    return ".cpp";
  case clang::Language::ObjC:
    return ".m";
  case clang::Language::ObjCXX:
    return ".mm";
  }
}

std::unique_ptr<MemoryBuffer> createInputBuffer(const InstallAPIContext &Ctx) {
  assert(Ctx.Type != HeaderType::Unknown &&
         "unexpected access level for parsing");
  SmallString<4096> Contents;
  raw_svector_ostream OS(Contents);
  for (const HeaderFile &H : Ctx.InputHeaders) {
    if (H.getType() != Ctx.Type)
      continue;
    if (Ctx.LangMode == Language::C || Ctx.LangMode == Language::CXX)
      OS << "#include ";
    else
      OS << "#import ";
    if (H.useIncludeName())
      OS << "<" << H.getIncludeName() << ">";
    else
      OS << "\"" << H.getPath() << "\"";
  }
  if (Contents.empty())
    return nullptr;

  SmallString<64> BufferName(
      {"installapi-includes-", Ctx.Slice->getTriple().str(), "-",
       getName(Ctx.Type), getFileExtension(Ctx.LangMode)});
  return llvm::MemoryBuffer::getMemBufferCopy(Contents, BufferName);
}

} // namespace clang::installapi
