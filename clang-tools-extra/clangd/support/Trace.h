//===--- Trace.h - Performance tracing facilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_TRACE_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_TRACE_H_

#include "support/Context.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace trace {

/// Represents measurements of clangd events, e.g. operation latency. Those
/// measurements are recorded per-label, defaulting to an empty one for metrics
/// that don't care about it. This enables aggregation of measurements across
/// labels. For example a metric tracking accesses to a cache can have labels
/// named hit and miss.
struct Metric {
  enum MetricType {
    /// A number whose value is meaningful, and may vary over time.
    /// Each measurement replaces the current value.
    Value,

    /// An aggregate number whose rate of change over time is meaningful.
    /// Each measurement is an increment for the counter.
    Counter,

    /// A distribution of values with a meaningful mean and count.
    /// Each measured value is a sample for the distribution.
    /// The distribution is assumed not to vary, samples are aggregated over
    /// time.
    Distribution,
  };
  constexpr Metric(llvm::StringLiteral Name, MetricType Type,
                   llvm::StringLiteral LabelName = llvm::StringLiteral(""))
      : Name(Name), Type(Type), LabelName(LabelName) {}

  /// Records a measurement for this metric to active tracer.
  void record(double Value, llvm::StringRef Label = "") const;

  /// Uniquely identifies the metric. Should use snake_case identifiers, can use
  /// dots for hierarchy if needed. e.g. method_latency, foo.bar.
  const llvm::StringLiteral Name;
  const MetricType Type;
  /// Indicates what measurement labels represent, e.g. "operation_name" for a
  /// metric tracking latencies. If non empty all measurements must also have a
  /// non-empty label.
  const llvm::StringLiteral LabelName;
};

/// A consumer of trace events and measurements. The events are produced by
/// Spans and trace::log, the measurements are produced by Metrics::record.
/// Implementations of this interface must be thread-safe.
class EventTracer {
public:
  virtual ~EventTracer() = default;

  /// Called when event that has a duration starts. \p Name describes the event.
  /// Returns a derived context that will be destroyed when the event ends.
  /// Usually implementations will store an object in the returned context
  /// whose destructor records the end of the event.
  /// The tracer may capture event details provided in SPAN_ATTACH() calls.
  /// In this case it should call AttachDetails(), and pass in an empty Object
  /// to hold them. This Object should be owned by the context, and the data
  /// will be complete by the time the context is destroyed.
  virtual Context
  beginSpan(llvm::StringRef Name,
            llvm::function_ref<void(llvm::json::Object *)> AttachDetails);
  // Called when a Span is destroyed (it may still be active on other threads).
  // beginSpan() and endSpan() will always form a proper stack on each thread.
  // The Context returned by beginSpan is active, but Args is not ready.
  // Tracers should not override this unless they need to observe strict
  // per-thread nesting. Instead they should observe context destruction.
  virtual void endSpan() {}

  /// Called for instant events.
  virtual void instant(llvm::StringRef Name, llvm::json::Object &&Args) {}

  /// Called whenever a metrics records a measurement.
  virtual void record(const Metric &Metric, double Value,
                      llvm::StringRef Label) {}
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
/// FIXME: Metrics are not recorded, some could become counter events.
///
/// The format is documented here:
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
std::unique_ptr<EventTracer> createJSONTracer(llvm::raw_ostream &OS,
                                              bool Pretty = false);

/// Create an instance of EventTracer that outputs metric measurements as CSV.
///
/// Trace spans and instant events are ignored.
std::unique_ptr<EventTracer> createCSVMetricTracer(llvm::raw_ostream &OS);

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
  /// Records span's duration in seconds to \p LatencyMetric with \p Name as the
  /// label.
  Span(llvm::Twine Name, const Metric &LatencyMetric);
  ~Span();

  /// Mutable metadata, if this span is interested.
  /// Prefer to use SPAN_ATTACH rather than accessing this directly.
  /// The lifetime of Args is the whole event, even if the Span dies.
  llvm::json::Object *const Args;

private:
  // Awkward constructor works around constant initialization.
  Span(std::pair<Context, llvm::json::Object *>);
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

#endif
