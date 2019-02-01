//===--- SyncAPI.cpp - Sync version of ClangdServer's API --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyncAPI.h"
#include "index/Index.h"

namespace clang {
namespace clangd {

void runAddDocument(ClangdServer &Server, PathRef File,
                    llvm::StringRef Contents, WantDiagnostics WantDiags) {
  Server.addDocument(File, Contents, WantDiags);
  if (!Server.blockUntilIdleForTest())
    llvm_unreachable("not idle after addDocument");
}

namespace {
/// A helper that waits for async callbacks to fire and exposes their result in
/// the output variable. Intended to be used in the following way:
///    T Result;
///    someAsyncFunc(Param1, Param2, /*Callback=*/capture(Result));
template <typename T> struct CaptureProxy {
  CaptureProxy(llvm::Optional<T> &Target) : Target(&Target) {
    assert(!Target.hasValue());
  }

  CaptureProxy(const CaptureProxy &) = delete;
  CaptureProxy &operator=(const CaptureProxy &) = delete;
  // We need move ctor to return a value from the 'capture' helper.
  CaptureProxy(CaptureProxy &&Other) : Target(Other.Target) {
    Other.Target = nullptr;
  }
  CaptureProxy &operator=(CaptureProxy &&) = delete;

  operator llvm::unique_function<void(T)>() && {
    assert(!Future.valid() && "conversion to callback called multiple times");
    Future = Promise.get_future();
    return Bind(
        [](std::promise<std::shared_ptr<T>> Promise, T Value) {
          Promise.set_value(std::make_shared<T>(std::move(Value)));
        },
        std::move(Promise));
  }

  ~CaptureProxy() {
    if (!Target)
      return;
    assert(Future.valid() && "conversion to callback was not called");
    assert(!Target->hasValue());
    Target->emplace(std::move(*Future.get()));
  }

private:
  llvm::Optional<T> *Target;
  // Using shared_ptr to workaround compilation errors with MSVC.
  // MSVC only allows default-construcitble and copyable objects as future<>
  // arguments.
  std::promise<std::shared_ptr<T>> Promise;
  std::future<std::shared_ptr<T>> Future;
};

template <typename T> CaptureProxy<T> capture(llvm::Optional<T> &Target) {
  return CaptureProxy<T>(Target);
}
} // namespace

llvm::Expected<CodeCompleteResult>
runCodeComplete(ClangdServer &Server, PathRef File, Position Pos,
                clangd::CodeCompleteOptions Opts) {
  llvm::Optional<llvm::Expected<CodeCompleteResult>> Result;
  Server.codeComplete(File, Pos, Opts, capture(Result));
  return std::move(*Result);
}

llvm::Expected<SignatureHelp> runSignatureHelp(ClangdServer &Server,
                                               PathRef File, Position Pos) {
  llvm::Optional<llvm::Expected<SignatureHelp>> Result;
  Server.signatureHelp(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<LocatedSymbol>>
runLocateSymbolAt(ClangdServer &Server, PathRef File, Position Pos) {
  llvm::Optional<llvm::Expected<std::vector<LocatedSymbol>>> Result;
  Server.locateSymbolAt(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<DocumentHighlight>>
runFindDocumentHighlights(ClangdServer &Server, PathRef File, Position Pos) {
  llvm::Optional<llvm::Expected<std::vector<DocumentHighlight>>> Result;
  Server.findDocumentHighlights(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<tooling::Replacement>>
runRename(ClangdServer &Server, PathRef File, Position Pos,
          llvm::StringRef NewName) {
  llvm::Optional<llvm::Expected<std::vector<tooling::Replacement>>> Result;
  Server.rename(File, Pos, NewName, capture(Result));
  return std::move(*Result);
}

std::string runDumpAST(ClangdServer &Server, PathRef File) {
  llvm::Optional<std::string> Result;
  Server.dumpAST(File, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<SymbolInformation>>
runWorkspaceSymbols(ClangdServer &Server, llvm::StringRef Query, int Limit) {
  llvm::Optional<llvm::Expected<std::vector<SymbolInformation>>> Result;
  Server.workspaceSymbols(Query, Limit, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<DocumentSymbol>>
runDocumentSymbols(ClangdServer &Server, PathRef File) {
  llvm::Optional<llvm::Expected<std::vector<DocumentSymbol>>> Result;
  Server.documentSymbols(File, capture(Result));
  return std::move(*Result);
}

SymbolSlab runFuzzyFind(const SymbolIndex &Index, llvm::StringRef Query) {
  FuzzyFindRequest Req;
  Req.Query = Query;
  Req.AnyScope = true;
  return runFuzzyFind(Index, Req);
}

SymbolSlab runFuzzyFind(const SymbolIndex &Index, const FuzzyFindRequest &Req) {
  SymbolSlab::Builder Builder;
  Index.fuzzyFind(Req, [&](const Symbol &Sym) { Builder.insert(Sym); });
  return std::move(Builder).build();
}

RefSlab getRefs(const SymbolIndex &Index, SymbolID ID) {
  RefsRequest Req;
  Req.IDs = {ID};
  RefSlab::Builder Slab;
  Index.refs(Req, [&](const Ref &S) { Slab.insert(ID, S); });
  return std::move(Slab).build();
}

} // namespace clangd
} // namespace clang
