// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.unix.BlockInCriticalSection -std=c++11 -verify %s

void sleep(int x) {}

namespace std {
struct mutex {
  void lock() {}
  void unlock() {}
};
}

void testBlockInCriticalSection() {
  std::mutex m;
  m.lock();
  sleep(3); // expected-warning {{A blocking function %s is called inside a critical section}}
  m.unlock();
}

void testBlockInCriticalSectionWithNestedMutexes() {
  std::mutex m, n, k;
  m.lock();
  n.lock();
  k.lock();
  sleep(3); // expected-warning {{A blocking function %s is called inside a critical section}}
  k.unlock();
  sleep(5); // expected-warning {{A blocking function %s is called inside a critical section}}
  n.unlock();
  sleep(3); // expected-warning {{A blocking function %s is called inside a critical section}}
  m.unlock();
  sleep(3); // no-warning
}

void f() {
  sleep(1000); // expected-warning {{A blocking function %s is called inside a critical section}}
}

void testBlockInCriticalSectionInterProcedural() {
  std::mutex m;
  m.lock();
  f();
  m.unlock();
}

void testBlockInCriticalSectionUnexpectedUnlock() {
  std::mutex m;
  m.unlock();
  sleep(1); // no-warning
  m.lock();
  sleep(1); // expected-warning {{A blocking function %s is called inside a critical section}}
}
