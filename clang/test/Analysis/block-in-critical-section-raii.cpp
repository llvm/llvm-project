// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix.BlockInCriticalSection

void sleep(int x);

namespace std {
struct mutex {
  void lock();
  void unlock();
};
// libc++-shaped unique_lock: ctor and dtor have non-empty bodies that call
// the underlying mutex's lock()/unlock() member functions.
template <class M>
struct unique_lock {
  M *m_;
  bool owns_;
  explicit unique_lock(M &m) : m_(&m), owns_(true) { m_->lock(); }
  ~unique_lock() {
    if (owns_)
      m_->unlock();
  }
};
} // namespace std

// Recursive use of the RAII guard. Without the fix, deep recursion would
// prevent the dtor body from inlining at the inlining-stack-depth limit;
// the inner mutex unlock would not fire, used to emit a leak FP.
struct C {
  std::mutex m;
  bool aborted;
  bool getAborted() {
    std::unique_lock<std::mutex> lk(m);
    return aborted;
  }
  void recurse() {
    if (getAborted())
      recurse();
    sleep(1); // no-warning
  }
};

// Direct case: RAII guard scoped to a function body. Constructor inlining +
// destructor inlining must not double-count, even when both bodies inline.
void callee(std::mutex &m) {
  std::unique_lock<std::mutex> lk(m);
  // The lock IS held here so blocking is correctly diagnosed.
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
} // dtor fires here, releases the lock; no further sleep should warn.

void top(std::mutex &m) {
  callee(m);
  sleep(1);  // no-warning: the lock from `callee` was released by ~unique_lock before this sleep.
}
