//===--- SyncAPI.cpp - Sync version of ClangdServer's API --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyncAPI.h"
#include "Protocol.h"
#include "index/Index.h"
#include <optional>

namespace clang {
namespace clangd {

void runAddDocument(ClangdServer &Server, PathRef File,
                    llvm::StringRef Contents, llvm::StringRef Version,
                    WantDiagnostics WantDiags, bool ForceRebuild) {
  Server.addDocument(File, Contents, Version, WantDiags, ForceRebuild);
  if (!Server.blockUntilIdleForTest())
    llvm_unreachable("not idle after addDocument");
}

namespace {
/// A helper that waits for async callbacks to fire and exposes their result in
/// the output variable. Intended to be used in the following way:
///    T Result;
///    someAsyncFunc(Param1, Param2, /*Callback=*/capture(Result));
template <typename T> struct CaptureProxy {
  CaptureProxy(std::optional<T> &Target) : Target(&Target) { assert(!Target); }

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
    return [Promise = std::move(Promise)](T Value) mutable {
      Promise.set_value(std::make_shared<T>(std::move(Value)));
    };
  }

  ~CaptureProxy() {
    if (!Target)
      return;
    assert(Future.valid() && "conversion to callback was not called");
    assert(!Target->has_value());
    Target->emplace(std::move(*Future.get()));
  }

private:
  std::optional<T> *Target;
  // Using shared_ptr to workaround compilation errors with MSVC.
  // MSVC only allows default-constructible and copyable objects as future<>
  // arguments.
  std::promise<std::shared_ptr<T>> Promise;
  std::future<std::shared_ptr<T>> Future;
};

template <typename T> CaptureProxy<T> capture(std::optional<T> &Target) {
  return CaptureProxy<T>(Target);
}
} // namespace

llvm::Expected<CodeCompleteResult>
runCodeComplete(ClangdServer &Server, PathRef File, Position Pos,
                clangd::CodeCompleteOptions Opts) {
  std::optional<llvm::Expected<CodeCompleteResult>> Result;
  Server.codeComplete(File, Pos, Opts, capture(Result));
  return std::move(*Result);
}

llvm::Expected<SignatureHelp> runSignatureHelp(ClangdServer &Server,
                                               PathRef File, Position Pos,
                                               MarkupKind DocumentationFormat) {
  std::optional<llvm::Expected<SignatureHelp>> Result;
  Server.signatureHelp(File, Pos, DocumentationFormat, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<LocatedSymbol>>
runLocateSymbolAt(ClangdServer &Server, PathRef File, Position Pos) {
  std::optional<llvm::Expected<std::vector<LocatedSymbol>>> Result;
  Server.locateSymbolAt(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::vector<DocumentHighlight>>
runFindDocumentHighlights(ClangdServer &Server, PathRef File, Position Pos) {
  std::optional<llvm::Expected<std::vector<DocumentHighlight>>> Result;
  Server.findDocumentHighlights(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<RenameResult> runRename(ClangdServer &Server, PathRef File,
                                       Position Pos, llvm::StringRef NewName,
                                       const RenameOptions &RenameOpts) {
  std::optional<llvm::Expected<RenameResult>> Result;
  Server.rename(File, Pos, NewName, RenameOpts, capture(Result));
  return std::move(*Result);
}

llvm::Expected<RenameResult>
runPrepareRename(ClangdServer &Server, PathRef File, Position Pos,
                 std::optional<std::string> NewName,
                 const RenameOptions &RenameOpts) {
  std::optional<llvm::Expected<RenameResult>> Result;
  Server.prepareRename(File, Pos, NewName, RenameOpts, capture(Result));
  return std::move(*Result);
}

llvm::Expected<tooling::Replacements>
runFormatFile(ClangdServer &Server, PathRef File,
              const std::vector<Range> &Rngs) {
  std::optional<llvm::Expected<tooling::Replacements>> Result;
  Server.formatFile(File, Rngs, capture(Result));
  return std::move(*Result);
}

SymbolSlab runFuzzyFind(const SymbolIndex &Index, llvm::StringRef Query) {
  FuzzyFindRequest Req;
  Req.Query = std::string(Query);
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

llvm::Expected<std::vector<SelectionRange>>
runSemanticRanges(ClangdServer &Server, PathRef File,
                  const std::vector<Position> &Pos) {
  std::optional<llvm::Expected<std::vector<SelectionRange>>> Result;
  Server.semanticRanges(File, Pos, capture(Result));
  return std::move(*Result);
}

llvm::Expected<std::optional<clangd::Path>>
runSwitchHeaderSource(ClangdServer &Server, PathRef File) {
  std::optional<llvm::Expected<std::optional<clangd::Path>>> Result;
  Server.switchSourceHeader(File, capture(Result));
  return std::move(*Result);
}

llvm::Error runCustomAction(ClangdServer &Server, PathRef File,
                            llvm::function_ref<void(InputsAndAST)> Action) {
  llvm::Error Result = llvm::Error::success();
  Notification Done;
  Server.customAction(File, "Custom", [&](llvm::Expected<InputsAndAST> AST) {
    if (!AST)
      Result = AST.takeError();
    else
      Action(*AST);
    Done.notify();
  });
  Done.wait();
  return Result;
}

} // namespace clangd
} // namespace clang
