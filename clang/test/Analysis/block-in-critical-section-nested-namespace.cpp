
// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

unsigned int sleep(unsigned int seconds) {return 0;}
namespace std {
namespace __detail {
class __mutex_base {
public:
  void lock();
};
} // namespace __detail

class mutex : public __detail::__mutex_base{
public:
  void unlock();
  bool try_lock();
};
} // namespace std

void gh_99628() {
  std::mutex m;
  m.lock();
  // expected-note@-1 {{Entering critical section here}}
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
}

void no_false_positive_gh_104241() {
  std::mutex m;
  m.lock();
  // If inheritance not handled properly, this unlock might not match the lock
  // above because technically they act on different memory regions:
  // __mutex_base and mutex.
  m.unlock();
  sleep(10); // no-warning
}
