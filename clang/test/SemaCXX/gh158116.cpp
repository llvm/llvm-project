// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -std=c++11 %s

class __attribute__((lockable)) Mutex {
public:
  void lock() __attribute__((exclusive_lock_function));
  void unlock() __attribute__((unlock_function));
};

class Foo {
  Mutex mu;
  int y __attribute__((guarded_by(mu)));
  int z [[clang::guarded_by(mu)]];

  void func1() {
    y = 0;  // expected-warning{{writing variable 'y' requires holding}}
    z = 1;  // expected-warning{{writing variable 'z' requires holding}}
  }

  void func2() {
    mu.lock();
    y = 2;
    z = 3;
    mu.unlock();
  }
};
