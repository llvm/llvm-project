//===-- sanitizer_stoptheworld_mac.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See sanitizer_stoptheworld.h for details.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_MAC && (defined(__x86_64__) || defined(__aarch64__) || \
                      defined(__i386))

#include <mach/mach.h>

#include "sanitizer_stoptheworld.h"

namespace __sanitizer {
typedef struct {
  tid_t tid;
  thread_t thread;
} SuspendedThreadInfo;

class SuspendedThreadsListMac : public SuspendedThreadsList {
 public:
  SuspendedThreadsListMac() : threads_(1024) {}

  tid_t GetThreadID(uptr index) const;
  thread_t GetThread(uptr index) const;
  uptr ThreadCount() const;
  bool ContainsThread(thread_t thread) const;
  void Append(thread_t thread);

  PtraceRegistersStatus GetRegistersAndSP(uptr index, uptr *buffer,
                                          uptr *sp) const;
  uptr RegisterCount() const;

 private:
  InternalMmapVector<SuspendedThreadInfo> threads_;
};

void StopTheWorld(StopTheWorldCallback callback, void *argument) {
  CHECK(0 && "unimplemented");
}

tid_t SuspendedThreadsListMac::GetThreadID(uptr index) const {
  CHECK_LT(index, threads_.size());
  return threads_[index].tid;
}

thread_t SuspendedThreadsListMac::GetThread(uptr index) const {
  CHECK_LT(index, threads_.size());
  return threads_[index].thread;
}

uptr SuspendedThreadsListMac::ThreadCount() const {
  return threads_.size();
}

bool SuspendedThreadsListMac::ContainsThread(thread_t thread) const {
  for (uptr i = 0; i < threads_.size(); i++) {
    if (threads_[i].thread == thread) return true;
  }
  return false;
}

void SuspendedThreadsListMac::Append(thread_t thread) {
  thread_identifier_info_data_t info;
  mach_msg_type_number_t info_count = THREAD_IDENTIFIER_INFO_COUNT;
  kern_return_t err = thread_info(thread, THREAD_IDENTIFIER_INFO,
                                  (thread_info_t)&info, &info_count);
  if (err != KERN_SUCCESS) {
    VReport(1, "Error - unable to get thread ident for a thread\n");
    return;
  }
  threads_.push_back({info.thread_id, thread});
}

PtraceRegistersStatus SuspendedThreadsListMac::GetRegistersAndSP(
    uptr index, uptr *buffer, uptr *sp) const {
  CHECK(0 && "unimplemented");
  return REGISTERS_UNAVAILABLE_FATAL;
}

uptr SuspendedThreadsListMac::RegisterCount() const {
  return MACHINE_THREAD_STATE_COUNT;
}
} // namespace __sanitizer

#endif  // SANITIZER_MAC && (defined(__x86_64__) || defined(__aarch64__)) ||
        //                   defined(__i386))
