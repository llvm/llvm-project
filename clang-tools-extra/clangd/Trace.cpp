//===--- Trace.cpp - Performance tracing facilities -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Trace.h"
#include "Context.h"
#include "Function.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include <atomic>
#include <mutex>

namespace clang {
namespace clangd {
namespace trace {

namespace {
// The current implementation is naive: each thread writes to Out guarded by Mu.
// Perhaps we should replace this by something that disturbs performance less.
class JSONTracer : public EventTracer {
public:
  JSONTracer(llvm::raw_ostream &Out, bool Pretty)
      : Out(Out), Sep(""), Start(std::chrono::system_clock::now()),
        JSONFormat(Pretty ? "{0:2}" : "{0}") {
    // The displayTimeUnit must be ns to avoid low-precision overlap
    // calculations!
    Out << R"({"displayTimeUnit":"ns","traceEvents":[)"
        << "\n";
    rawEvent("M", llvm::json::Object{
                      {"name", "process_name"},
                      {"args", llvm::json::Object{{"name", "clangd"}}},
                  });
  }

  ~JSONTracer() {
    Out << "\n]}";
    Out.flush();
  }

  // We stash a Span object in the context. It will record the start/end,
  // and this also allows us to look up the parent Span's information.
  Context beginSpan(llvm::StringRef Name, llvm::json::Object *Args) override {
    return Context::current().derive(
        SpanKey, llvm::make_unique<JSONSpan>(this, Name, Args));
  }

  // Trace viewer requires each thread to properly stack events.
  // So we need to mark only duration that the span was active on the thread.
  // (Hopefully any off-thread activity will be connected by a flow event).
  // Record the end time here, but don't write the event: Args aren't ready yet.
  void endSpan() override {
    Context::current().getExisting(SpanKey)->markEnded();
  }

  void instant(llvm::StringRef Name, llvm::json::Object &&Args) override {
    captureThreadMetadata();
    jsonEvent("i",
              llvm::json::Object{{"name", Name}, {"args", std::move(Args)}});
  }

  // Record an event on the current thread. ph, pid, tid, ts are set.
  // Contents must be a list of the other JSON key/values.
  void jsonEvent(llvm::StringRef Phase, llvm::json::Object &&Contents,
                 uint64_t TID = llvm::get_threadid(), double Timestamp = 0) {
    Contents["ts"] = Timestamp ? Timestamp : timestamp();
    Contents["tid"] = int64_t(TID);
    std::lock_guard<std::mutex> Lock(Mu);
    rawEvent(Phase, std::move(Contents));
  }

private:
  class JSONSpan {
  public:
    JSONSpan(JSONTracer *Tracer, llvm::StringRef Name, llvm::json::Object *Args)
        : StartTime(Tracer->timestamp()), EndTime(0), Name(Name),
          TID(llvm::get_threadid()), Tracer(Tracer), Args(Args) {
      // ~JSONSpan() may run in a different thread, so we need to capture now.
      Tracer->captureThreadMetadata();

      // We don't record begin events here (and end events in the destructor)
      // because B/E pairs have to appear in the right order, which is awkward.
      // Instead we send the complete (X) event in the destructor.

      // If our parent was on a different thread, add an arrow to this span.
      auto *Parent = Context::current().get(SpanKey);
      if (Parent && *Parent && (*Parent)->TID != TID) {
        // If the parent span ended already, then show this as "following" it.
        // Otherwise show us as "parallel".
        double OriginTime = (*Parent)->EndTime;
        if (!OriginTime)
          OriginTime = (*Parent)->StartTime;

        auto FlowID = nextID();
        Tracer->jsonEvent(
            "s",
            llvm::json::Object{{"id", FlowID},
                               {"name", "Context crosses threads"},
                               {"cat", "dummy"}},
            (*Parent)->TID, (*Parent)->StartTime);
        Tracer->jsonEvent(
            "f",
            llvm::json::Object{{"id", FlowID},
                               {"bp", "e"},
                               {"name", "Context crosses threads"},
                               {"cat", "dummy"}},
            TID);
      }
    }

    ~JSONSpan() {
      // Finally, record the event (ending at EndTime, not timestamp())!
      Tracer->jsonEvent("X",
                        llvm::json::Object{{"name", std::move(Name)},
                                           {"args", std::move(*Args)},
                                           {"dur", EndTime - StartTime}},
                        TID, StartTime);
    }

    // May be called by any thread.
    void markEnded() { EndTime = Tracer->timestamp(); }

  private:
    static int64_t nextID() {
      static std::atomic<int64_t> Next = {0};
      return Next++;
    }

    double StartTime;
    std::atomic<double> EndTime; // Filled in by markEnded().
    std::string Name;
    uint64_t TID;
    JSONTracer *Tracer;
    llvm::json::Object *Args;
  };
  static Key<std::unique_ptr<JSONSpan>> SpanKey;

  // Record an event. ph and pid are set.
  // Contents must be a list of the other JSON key/values.
  void rawEvent(llvm::StringRef Phase,
                llvm::json::Object &&Event) /*REQUIRES(Mu)*/ {
    // PID 0 represents the clangd process.
    Event["pid"] = 0;
    Event["ph"] = Phase;
    Out << Sep
        << llvm::formatv(JSONFormat, llvm::json::Value(std::move(Event)));
    Sep = ",\n";
  }

  // If we haven't already, emit metadata describing this thread.
  void captureThreadMetadata() {
    uint64_t TID = llvm::get_threadid();
    std::lock_guard<std::mutex> Lock(Mu);
    if (ThreadsWithMD.insert(TID).second) {
      llvm::SmallString<32> Name;
      llvm::get_thread_name(Name);
      if (!Name.empty()) {
        rawEvent("M", llvm::json::Object{
                          {"tid", int64_t(TID)},
                          {"name", "thread_name"},
                          {"args", llvm::json::Object{{"name", Name}}},
                      });
      }
    }
  }

  double timestamp() {
    using namespace std::chrono;
    return duration<double, std::micro>(system_clock::now() - Start).count();
  }

  std::mutex Mu;
  llvm::raw_ostream &Out /*GUARDED_BY(Mu)*/;
  const char *Sep /*GUARDED_BY(Mu)*/;
  llvm::DenseSet<uint64_t> ThreadsWithMD /*GUARDED_BY(Mu)*/;
  const llvm::sys::TimePoint<> Start;
  const char *JSONFormat;
};

Key<std::unique_ptr<JSONTracer::JSONSpan>> JSONTracer::SpanKey;

EventTracer *T = nullptr;
} // namespace

Session::Session(EventTracer &Tracer) {
  assert(!T && "Resetting global tracer is not allowed.");
  T = &Tracer;
}

Session::~Session() { T = nullptr; }

std::unique_ptr<EventTracer> createJSONTracer(llvm::raw_ostream &OS,
                                              bool Pretty) {
  return llvm::make_unique<JSONTracer>(OS, Pretty);
}

void log(const llvm::Twine &Message) {
  if (!T)
    return;
  T->instant("Log", llvm::json::Object{{"Message", Message.str()}});
}

// Returned context owns Args.
static Context makeSpanContext(llvm::Twine Name, llvm::json::Object *Args) {
  if (!T)
    return Context::current().clone();
  WithContextValue WithArgs{std::unique_ptr<llvm::json::Object>(Args)};
  return T->beginSpan(Name.isSingleStringRef() ? Name.getSingleStringRef()
                                               : llvm::StringRef(Name.str()),
                      Args);
}

// Span keeps a non-owning pointer to the args, which is how users access them.
// The args are owned by the context though. They stick around until the
// beginSpan() context is destroyed, when the tracing engine will consume them.
Span::Span(llvm::Twine Name)
    : Args(T ? new llvm::json::Object() : nullptr),
      RestoreCtx(makeSpanContext(Name, Args)) {}

Span::~Span() {
  if (T)
    T->endSpan();
}

} // namespace trace
} // namespace clangd
} // namespace clang
