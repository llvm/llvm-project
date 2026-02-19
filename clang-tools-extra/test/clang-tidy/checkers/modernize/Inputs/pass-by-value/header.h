class ThreadId {
public:
  ThreadId(const ThreadId &) {}
  ThreadId(ThreadId &&) {}
};

struct A {
  A(const ThreadId &tid) : threadid(tid) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: pass by value and use std::move [modernize-pass-by-value]
  // CHECK-FIXES: #include <utility>
  // CHECK-FIXES: A(ThreadId tid) : threadid(std::move(tid)) {}
  ThreadId threadid;
};
