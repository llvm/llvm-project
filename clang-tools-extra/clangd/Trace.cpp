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
#include <mutex>

namespace clang {
namespace clangd {
namespace trace {
using namespace llvm;

namespace {
// The current implementation is naive: each thread writes to Out guarded by Mu.
// Perhaps we should replace this by something that disturbs performance less.
class JSONTracer : public EventTracer {
public:
  JSONTracer(raw_ostream &Out, bool Pretty)
      : Out(Out), Sep(""), Start(std::chrono::system_clock::now()),
        JSONFormat(Pretty ? "{0:2}" : "{0}") {
    // The displayTimeUnit must be ns to avoid low-precision overlap
    // calculations!
    Out << R"({"displayTimeUnit":"ns","traceEvents":[)"
        << "\n";
    rawEvent("M", json::obj{
                      {"name", "process_name"},
                      {"args", json::obj{{"name", "clangd"}}},
                  });
  }

  ~JSONTracer() {
    Out << "\n]}";
    Out.flush();
  }

  Context beginSpan(llvm::StringRef Name, json::obj *Args) override {
    jsonEvent("B", json::obj{{"name", Name}});
    return Context::current().derive(make_scope_exit([this, Args] {
      jsonEvent("E", json::obj{{"args", std::move(*Args)}});
    }));
  }

  void instant(llvm::StringRef Name, json::obj &&Args) override {
    jsonEvent("i", json::obj{{"name", Name}, {"args", std::move(Args)}});
  }

  // Record an event on the current thread. ph, pid, tid, ts are set.
  // Contents must be a list of the other JSON key/values.
  void jsonEvent(StringRef Phase, json::obj &&Contents) {
    uint64_t TID = get_threadid();
    std::lock_guard<std::mutex> Lock(Mu);
    // If we haven't already, emit metadata describing this thread.
    if (ThreadsWithMD.insert(TID).second) {
      SmallString<32> Name;
      get_thread_name(Name);
      if (!Name.empty()) {
        rawEvent("M", json::obj{
                          {"tid", TID},
                          {"name", "thread_name"},
                          {"args", json::obj{{"name", Name}}},
                      });
      }
    }
    Contents["ts"] = timestamp();
    Contents["tid"] = TID;
    rawEvent(Phase, std::move(Contents));
  }

private:
  // Record an event. ph and pid are set.
  // Contents must be a list of the other JSON key/values.
  void rawEvent(StringRef Phase, json::obj &&Event) /*REQUIRES(Mu)*/ {
    // PID 0 represents the clangd process.
    Event["pid"] = 0;
    Event["ph"] = Phase;
    Out << Sep << formatv(JSONFormat, json::Expr(std::move(Event)));
    Sep = ",\n";
  }

  double timestamp() {
    using namespace std::chrono;
    return duration<double, std::micro>(system_clock::now() - Start).count();
  }

  std::mutex Mu;
  raw_ostream &Out /*GUARDED_BY(Mu)*/;
  const char *Sep /*GUARDED_BY(Mu)*/;
  DenseSet<uint64_t> ThreadsWithMD /*GUARDED_BY(Mu)*/;
  const sys::TimePoint<> Start;
  const char *JSONFormat;
};

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

void log(const Twine &Message) {
  if (!T)
    return;
  T->instant("Log", json::obj{{"Message", Message.str()}});
}

// Returned context owns Args.
static Context makeSpanContext(llvm::StringRef Name, json::obj *Args) {
  if (!T)
    return Context::current().clone();
  WithContextValue WithArgs{std::unique_ptr<json::obj>(Args)};
  return T->beginSpan(Name, Args);
}

// Span keeps a non-owning pointer to the args, which is how users access them.
// The args are owned by the context though. They stick around until the
// beginSpan() context is destroyed, when the tracing engine will consume them.
Span::Span(llvm::StringRef Name)
    : Args(T ? new json::obj() : nullptr),
      RestoreCtx(makeSpanContext(Name, Args)) {}

} // namespace trace
} // namespace clangd
} // namespace clang
