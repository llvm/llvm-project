// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

unsigned int sleep(unsigned int seconds) {return 0;}
namespace std {
// There are some standard library implementations where some mutex methods
// come from an implementation detail base class. We need to ensure that these
// are matched correctly.
class __mutex_base {
public:
  void lock();
};
class mutex : public __mutex_base{
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
