//===--- SyncAPI.cpp - Sync version of ClangdServer's API --------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "SyncAPI.h"

namespace clang {
namespace clangd {

namespace {
/// A helper that waits for async callbacks to fire and exposes their result in
/// the output variable. Intended to be used in the following way:
///    T Result;
///    someAsyncFunc(Param1, Param2, /*Callback=*/capture(Result));
template <typename T> struct CaptureProxy {
  CaptureProxy(T &Target) : Target(&Target) {}

  CaptureProxy(const CaptureProxy &) = delete;
  CaptureProxy &operator=(const CaptureProxy &) = delete;
  // We need move ctor to return a value from the 'capture' helper.
  CaptureProxy(CaptureProxy &&Other) : Target(Other.Target) {
    Other.Target = nullptr;
  }
  CaptureProxy &operator=(CaptureProxy &&) = delete;

  operator UniqueFunction<void(T)>() && {
    assert(!Future.valid() && "conversion to callback called multiple times");
    Future = Promise.get_future();
    return BindWithForward([](std::promise<T> Promise,
                              T Value) { Promise.set_value(std::move(Value)); },
                           std::move(Promise));
  }

  ~CaptureProxy() {
    if (!Target)
      return;
    assert(Future.valid() && "conversion to callback was not called");
    *Target = Future.get();
  }

private:
  T *Target;
  std::promise<T> Promise;
  std::future<T> Future;
};

template <typename T> CaptureProxy<T> capture(T &Target) {
  return CaptureProxy<T>(Target);
}
} // namespace

Tagged<CompletionList>
runCodeComplete(ClangdServer &Server, PathRef File, Position Pos,
                clangd::CodeCompleteOptions Opts,
                llvm::Optional<StringRef> OverridenContents) {
  Tagged<CompletionList> Result;
  Server.codeComplete(File, Pos, Opts, capture(Result), OverridenContents);
  return Result;
}

} // namespace clangd
} // namespace clang
