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
#include "Function.h"
#include "JSONExpr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace trace {

/// A consumer of trace events. The events are produced by Spans and trace::log.
/// Implmentations of this interface must be thread-safe.
class EventTracer {
public:
  virtual ~EventTracer() = default;

  /// Called when event that has a duration starts. \p Name describes the event.
  /// Returns a derived context that will be destroyed when the event ends.
  /// Usually implementations will store an object in the returned context
  /// whose destructor records the end of the event.
  /// The args are *Args, only complete when the event ends.
  virtual Context beginSpan(llvm::StringRef Name, json::obj *Args) = 0;
  // Called when a Span is destroyed (it may still be active on other threads).
  // beginSpan() and endSpan() will always form a proper stack on each thread.
  // The Context returned by beginSpan is active, but Args is not ready.
  // Tracers should not override this unless they need to observe strict
  // per-thread nesting. Instead they should observe context destruction.
  virtual void endSpan(){};

  /// Called for instant events.
  virtual void instant(llvm::StringRef Name, json::obj &&Args) = 0;
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
std::unique_ptr<EventTracer> createJSONTracer(llvm::raw_ostream &OS,
                                              bool Pretty = false);

/// Records a single instant event, associated with the current thread.
void log(const llvm::Twine &Name);

/// Records an event whose duration is the lifetime of the Span object.
/// This lifetime is extended when the span's context is reused.
///
/// This is the main public interface for producing tracing events.
///
/// Arbitrary JSON metadata can be attached while this span is active:
///   SPAN_ATTACH(MySpan, "Payload", SomeJSONExpr);
///
/// SomeJSONExpr is evaluated and copied only if actually needed.
class Span {
public:
  Span(llvm::Twine Name);
  ~Span();

  /// Mutable metadata, if this span is interested.
  /// Prefer to use SPAN_ATTACH rather than accessing this directly.
  json::obj *const Args;

private:
  WithContext RestoreCtx;
};

/// Attach a key-value pair to a Span event.
/// This is not threadsafe when used with the same Span.
#define SPAN_ATTACH(S, Name, Expr)                                             \
  do {                                                                         \
    if (auto *Args = (S).Args)                                                 \
      (*Args)[Name] = Expr;                                                    \
  } while (0)

} // namespace trace
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TRACE_H_
