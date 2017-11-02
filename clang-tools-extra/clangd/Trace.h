//===--- Trace.h - Performance tracing facilities ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Supports writing performance traces describing clangd's behavior.
// Traces are written in the Trace Event format supported by chrome's trace
// viewer (chrome://tracing).
//
// The format is documented here:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
//
// All APIs are no-ops unless a Session is active (created by ClangdMain).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace trace {

// A session directs the output of trace events. Only one Session can exist.
// It should be created before clangd threads are spawned, and destroyed after
// they exit.
// TODO: we may want to add pluggable support for other tracing backends.
class Session {
public:
  // Starts a sessions capturing trace events and writing Trace Event JSON.
  static std::unique_ptr<Session> create(llvm::raw_ostream &OS);
  ~Session();

private:
  Session() = default;
};

// Records a single instant event, associated with the current thread.
void log(const llvm::Twine &Name);

// Records an event whose duration is the lifetime of the Span object.
class Span {
public:
  Span(const llvm::Twine &Name);
  ~Span();

private:
};

} // namespace trace
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_
