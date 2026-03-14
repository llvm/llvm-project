// RUN: %clang_cc1 %s -Wthread-safety-analysis -verify -fexceptions
// expected-no-diagnostics

class Mutex {
public:
  void Lock() __attribute__((exclusive_lock_function()));
  void Unlock() __attribute__((unlock_function()));
};

class A {
public:
  Mutex mu1, mu2;

  void foo() __attribute__((exclusive_locks_required(mu1))) __attribute__((exclusive_locks_required(mu2))) {}

  template <class T>
  void bar() __attribute__((exclusive_locks_required(mu1))) __attribute__((exclusive_locks_required(mu2))) {
    foo();
  }
};

void f() {
  A a;
  a.mu1.Lock();
  a.mu2.Lock();
  a.bar<int>();
  a.mu2.Unlock();
  a.mu1.Unlock();
}
