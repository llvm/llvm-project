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
  Tracer(raw_ostream &Out)
      : Out(Out), Sep(""), Start(std::chrono::system_clock::now()) {
    // The displayTimeUnit must be ns to avoid low-precision overlap
    // calculations!
    Out << R"({"displayTimeUnit":"ns","traceEvents":[)"
        << "\n";
    rawEvent("M", R"("name": "process_name", "args":{"name":"clangd"})");
  }

  ~Tracer() {
    Out << "\n]}";
    Out.flush();
  }

  // Record an event on the current thread. ph, pid, tid, ts are set.
  // Contents must be a list of the other JSON key/values.
  template <typename T> void event(StringRef Phase, const T &Contents) {
    uint64_t TID = get_threadid();
    std::lock_guard<std::mutex> Lock(Mu);
    // If we haven't already, emit metadata describing this thread.
    if (ThreadsWithMD.insert(TID).second) {
      SmallString<32> Name;
      get_thread_name(Name);
      if (!Name.empty()) {
        rawEvent(
            "M",
            formatv(
                R"("tid": {0}, "name": "thread_name", "args":{"name":"{1}"})",
                TID, StringRef(&Name[0], Name.size())));
      }
    }
    rawEvent(Phase, formatv(R"("ts":{0}, "tid":{1}, {2})", timestamp(), TID,
                            Contents));
  }

private:
  // Record an event. ph and pid are set.
  // Contents must be a list of the other JSON key/values.
  template <typename T>
  void rawEvent(StringRef Phase, const T &Contents) /*REQUIRES(Mu)*/ {
    // PID 0 represents the clangd process.
    Out << Sep << R"({"pid":0, "ph":")" << Phase << "\", " << Contents << "}";
    Sep = ",\n";
  }

  double timestamp() {
    using namespace std::chrono;
    return duration<double, std::milli>(system_clock::now() - Start).count();
  }

  std::mutex Mu;
  raw_ostream &Out /*GUARDED_BY(Mu)*/;
  const char *Sep /*GUARDED_BY(Mu)*/;
  DenseSet<uint64_t> ThreadsWithMD /*GUARDED_BY(Mu)*/;
  const sys::TimePoint<> Start;
};

static Tracer *T = nullptr;
} // namespace

std::unique_ptr<Session> Session::create(raw_ostream &OS) {
  assert(!T && "A session is already active");
  T = new Tracer(OS);
  return std::unique_ptr<Session>(new Session());
}

Session::~Session() {
  delete T;
  T = nullptr;
}

void log(const Twine &Message) {
  if (!T)
    return;
  T->event("i", formatv(R"("name":"{0}")", yaml::escape(Message.str())));
}

Span::Span(const Twine &Text) {
  if (!T)
    return;
  T->event("B", formatv(R"("name":"{0}")", yaml::escape(Text.str())));
}

Span::~Span() {
  if (!T)
    return;
  T->event("E", R"("_":0)" /* Dummy property to ensure valid JSON */);
}

} // namespace trace
} // namespace clangd
} // namespace clang
