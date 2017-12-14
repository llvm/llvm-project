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
// Traces are consumed by implementations of the EventTracer interface.
//
//
// All APIs are no-ops unless a Session is active (created by ClangdMain).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_

#include "Context.h"
#include "JSONExpr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace trace {

/// A consumer of trace events. The events are produced by Spans and trace::log.
class EventTracer {
public:
  virtual ~EventTracer() = default;
  /// Consume a trace event.
  virtual void event(const Context &Ctx, llvm::StringRef Phase,
                     json::obj &&Contents) = 0;
};

/// Sets up a global EventTracer that consumes events produced by Span and
/// trace::log. Only one TracingSession can be active at a time and it should be
/// set up before calling any clangd-specific functions.
class Session {
public:
  Session(EventTracer &Tracer);
  ~Session();
};

/// Create an instance of EventTracer that produces an output in the Trace Event
/// format supported by Chrome's trace viewer (chrome://tracing).
///
/// The format is documented here:
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
///
/// The implementation supports concurrent calls and can be used as a global
/// tracer (i.e., can be put into a global Context).
std::unique_ptr<EventTracer> createJSONTracer(llvm::raw_ostream &OS,
                                              bool Pretty = false);

/// Records a single instant event, associated with the current thread.
void log(const Context &Ctx, const llvm::Twine &Name);

/// Records an event whose duration is the lifetime of the Span object.
/// This is the main public interface for producing tracing events.
///
/// Arbitrary JSON metadata can be attached while this span is active:
///   SPAN_ATTACH(MySpan, "Payload", SomeJSONExpr);
/// SomeJSONExpr is evaluated and copied only if actually needed.
class Span {
public:
  Span(const Context &Ctx, std::string Name);
  ~Span();

  /// Returns mutable span metadata if this span is interested.
  /// Prefer to use SPAN_ATTACH rather than accessing this directly.
  json::obj *args() { return Args.get(); }

private:
  llvm::Optional<Context> Ctx;
  std::unique_ptr<json::obj> Args;
};

#define SPAN_ATTACH(S, Name, Expr)                                             \
  do {                                                                         \
    if ((S).args() != nullptr) {                                               \
      (*((S).args()))[(Name)] = (Expr);                                        \
    }                                                                          \
  } while (0)

} // namespace trace
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_
