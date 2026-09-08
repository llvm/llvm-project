//===--- PathIdentity.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/PathIdentity.h"
#include "URI.h"
#include "support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include <utility>

namespace clang {
namespace clangd {
namespace {

bool hasURIScheme(llvm::StringRef S) {
  if (S.empty() || !llvm::isAlpha(S.front()))
    return false;
  size_t Colon = S.find(':');
  if (Colon == llvm::StringRef::npos)
    return false;
  return llvm::all_of(S.take_front(Colon).drop_front(), [](char C) {
    return llvm::isAlnum(C) || C == '+' || C == '-' || C == '.';
  });
}

} // namespace

std::optional<IndexFileKeyRef>
indexFileIdentity(llvm::StringRef URIOrPath,
                  llvm::SmallVectorImpl<char> &Storage) {
  // A drive path has URI-like syntax ("c:"), so recognize it first. Inputs
  // without a valid URI scheme are unambiguously filesystem paths, including
  // relative and UNC paths.
  if (hasWindowsDrive(URIOrPath) || !hasURIScheme(URIOrPath))
    return IndexFileKeyRef{URIOrPath, IndexFileKeyRef::FilePath};

  if (!URIOrPath.starts_with("file:"))
    return IndexFileKeyRef{URIOrPath, IndexFileKeyRef::OpaqueURI};

  // Mirror the file scheme's authority/body handling without constructing a
  // URI or an owned path on every index coverage query. Escapes use the parser.
  if (!URIOrPath.contains('%')) {
    llvm::StringRef Body = URIOrPath.drop_front(5);
    bool HasAuthority = false;
    if (Body.starts_with("//")) {
      auto AuthorityAndBody = Body.drop_front(2);
      HasAuthority = !AuthorityAndBody.starts_with("/");
      if (!HasAuthority)
        Body = AuthorityAndBody;
      else if (!AuthorityAndBody.contains('/'))
        Body = {}; // Missing absolute body: let the resolver diagnose it.
    }
    if (Body.starts_with("/")) {
      if (!HasAuthority && hasWindowsDrive(Body.drop_front()))
        Body = Body.drop_front();
      return IndexFileKeyRef{Body, IndexFileKeyRef::FilePath};
    }
  }

  auto Parsed = URI::parse(URIOrPath);
  if (!Parsed) {
    elog("Invalid index file URI {0}: {1}", URIOrPath, Parsed.takeError());
    return std::nullopt;
  }
  auto Abs = URI::resolve(*Parsed, /*HintPath=*/"");
  if (!Abs) {
    elog("Failed to resolve index file URI {0}: {1}", URIOrPath,
         Abs.takeError());
    return std::nullopt;
  }
  Storage.assign(Abs->begin(), Abs->end());
  return IndexFileKeyRef{llvm::StringRef(Storage.data(), Storage.size()),
                         IndexFileKeyRef::FilePath};
}

std::optional<IndexFileKey> indexFileIdentity(llvm::StringRef URIOrPath) {
  llvm::SmallString<256> Storage;
  if (auto Ref = indexFileIdentity(URIOrPath, Storage))
    return IndexFileKey(*Ref);
  return std::nullopt;
}

} // namespace clangd
} // namespace clang
