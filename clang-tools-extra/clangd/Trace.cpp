//===--- Trace.cpp - Performance tracing facilities -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Trace.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include <mutex>

namespace clang {
namespace clangd {
namespace trace {
using namespace llvm;

namespace {
// The current implementation is naive: each thread writes to Out guarded by Mu.
// Perhaps we should replace this by something that disturbs performance less.
class Tracer {
public:
  Tracer(raw_ostream &Out, bool Pretty)
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

  ~Tracer() {
    Out << "\n]}";
    Out.flush();
  }

  // Record an event on the current thread. ph, pid, tid, ts are set.
  // Contents must be a list of the other JSON key/values.
  void event(StringRef Phase, json::obj &&Contents) {
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

static Tracer *T = nullptr;
} // namespace

std::unique_ptr<Session> Session::create(raw_ostream &OS, bool Pretty) {
  assert(!T && "A session is already active");
  T = new Tracer(OS, Pretty);
  return std::unique_ptr<Session>(new Session());
}

Session::~Session() {
  delete T;
  T = nullptr;
}

void log(const Twine &Message) {
  if (!T)
    return;
  T->event("i", json::obj{
                    {"name", "Log"},
                    {"args", json::obj{{"Message", Message.str()}}},
                });
}

Span::Span(std::string Name) {
  if (!T)
    return;
  T->event("B", json::obj{{"name", std::move(Name)}});
  Args = llvm::make_unique<json::obj>();
}

Span::~Span() {
  if (!T)
    return;
  if (!Args)
    Args = llvm::make_unique<json::obj>();
  T->event("E", Args ? json::obj{{"args", std::move(*Args)}} : json::obj{});
}

} // namespace trace
} // namespace clangd
} // namespace clang
