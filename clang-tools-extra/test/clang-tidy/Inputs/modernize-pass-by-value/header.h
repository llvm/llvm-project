class ThreadId {
public:
  ThreadId(const ThreadId &) {}
  ThreadId(ThreadId &&) {}
};

struct A {
  A(const ThreadId &tid) : threadid(tid) {}
  ThreadId threadid;
};
