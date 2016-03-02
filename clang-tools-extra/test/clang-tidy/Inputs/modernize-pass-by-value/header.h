class ThreadId {
};

struct A {
  A(const ThreadId &tid) : threadid(tid) {}
  ThreadId threadid;
};
