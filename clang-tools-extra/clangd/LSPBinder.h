//===--- LSPBinder.h - Tables of LSP handlers --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_LSPBINDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_LSPBINDER_H

#include "Protocol.h"
#include "support/Function.h"
#include "support/Logger.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

namespace clang {
namespace clangd {

/// LSPBinder collects a table of functions that handle LSP calls.
///
/// It translates a handler method's signature, e.g.
///    void codeComplete(CompletionParams, Callback<CompletionList>)
/// into a wrapper with a generic signature:
///    void(json::Value, Callback<json::Value>)
/// The wrapper takes care of parsing/serializing responses.
///
/// Incoming calls can be methods, notifications, or commands - all are similar.
///
/// FIXME: this should also take responsibility for wrapping *outgoing* calls,
/// replacing the typed ClangdLSPServer::call<> etc.
class LSPBinder {
public:
  using JSON = llvm::json::Value;

  struct RawHandlers {
    template <typename HandlerT>
    using HandlerMap = llvm::StringMap<llvm::unique_function<HandlerT>>;

    HandlerMap<void(JSON)> NotificationHandlers;
    HandlerMap<void(JSON, Callback<JSON>)> MethodHandlers;
    HandlerMap<void(JSON, Callback<JSON>)> CommandHandlers;
  };

  LSPBinder(RawHandlers &Raw) : Raw(Raw) {}

  /// Bind a handler for an LSP method.
  /// e.g. Bind.method("peek", this, &ThisModule::peek);
  /// Handler should be e.g. void peek(const PeekParams&, Callback<PeekResult>);
  /// PeekParams must be JSON-parseable and PeekResult must be serializable.
  template <typename Param, typename Result, typename ThisT>
  void method(llvm::StringLiteral Method, ThisT *This,
              void (ThisT::*Handler)(const Param &, Callback<Result>));

  /// Bind a handler for an LSP notification.
  /// e.g. Bind.notification("poke", this, &ThisModule::poke);
  /// Handler should be e.g. void poke(const PokeParams&);
  /// PokeParams must be JSON-parseable.
  template <typename Param, typename ThisT>
  void notification(llvm::StringLiteral Method, ThisT *This,
                    void (ThisT::*Handler)(const Param &));

  /// Bind a handler for an LSP command.
  /// e.g. Bind.command("load", this, &ThisModule::load);
  /// Handler should be e.g. void load(const LoadParams&, Callback<LoadResult>);
  /// LoadParams must be JSON-parseable and LoadResult must be serializable.
  template <typename Param, typename Result, typename ThisT>
  void command(llvm::StringLiteral Command, ThisT *This,
               void (ThisT::*Handler)(const Param &, Callback<Result>));

  // FIXME: remove usage from ClangdLSPServer and make this private.
  template <typename T>
  static llvm::Expected<T> parse(const llvm::json::Value &Raw,
                                 llvm::StringRef PayloadName,
                                 llvm::StringRef PayloadKind);

private:
  RawHandlers &Raw;
};

template <typename T>
llvm::Expected<T> LSPBinder::parse(const llvm::json::Value &Raw,
                                   llvm::StringRef PayloadName,
                                   llvm::StringRef PayloadKind) {
  T Result;
  llvm::json::Path::Root Root;
  if (!fromJSON(Raw, Result, Root)) {
    elog("Failed to decode {0} {1}: {2}", PayloadName, PayloadKind,
         Root.getError());
    // Dump the relevant parts of the broken message.
    std::string Context;
    llvm::raw_string_ostream OS(Context);
    Root.printErrorContext(Raw, OS);
    vlog("{0}", OS.str());
    // Report the error (e.g. to the client).
    return llvm::make_error<LSPError>(
        llvm::formatv("failed to decode {0} {1}: {2}", PayloadName, PayloadKind,
                      fmt_consume(Root.getError())),
        ErrorCode::InvalidParams);
  }
  return std::move(Result);
}

template <typename Param, typename Result, typename ThisT>
void LSPBinder::method(llvm::StringLiteral Method, ThisT *This,
                       void (ThisT::*Handler)(const Param &,
                                              Callback<Result>)) {
  Raw.MethodHandlers[Method] = [Method, Handler, This](JSON RawParams,
                                                       Callback<JSON> Reply) {
    auto P = LSPBinder::parse<Param>(RawParams, Method, "request");
    if (!P)
      return Reply(P.takeError());
    (This->*Handler)(*P, std::move(Reply));
  };
}

template <typename Param, typename ThisT>
void LSPBinder::notification(llvm::StringLiteral Method, ThisT *This,
                             void (ThisT::*Handler)(const Param &)) {
  Raw.NotificationHandlers[Method] = [Method, Handler, This](JSON RawParams) {
    llvm::Expected<Param> P =
        LSPBinder::parse<Param>(RawParams, Method, "request");
    if (!P)
      return llvm::consumeError(P.takeError());
    (This->*Handler)(*P);
  };
}

template <typename Param, typename Result, typename ThisT>
void LSPBinder::command(llvm::StringLiteral Method, ThisT *This,
                        void (ThisT::*Handler)(const Param &,
                                               Callback<Result>)) {
  Raw.CommandHandlers[Method] = [Method, Handler, This](JSON RawParams,
                                                        Callback<JSON> Reply) {
    auto P = LSPBinder::parse<Param>(RawParams, Method, "command");
    if (!P)
      return Reply(P.takeError());
    (This->*Handler)(*P, std::move(Reply));
  };
}

} // namespace clangd
} // namespace clang

#endif
