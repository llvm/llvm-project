// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wthread-safety-verbose -Wno-thread-safety-negative -fcxx-exceptions -DUSE_CAPABILITY=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wthread-safety-verbose -Wno-thread-safety-negative -fcxx-exceptions -DUSE_CAPABILITY=1 %s

#include "thread-safety-annotations.h"

class LOCKABLE Mutex {
 public:
  void Lock() EXCLUSIVE_LOCK_FUNCTION();
  void ReaderLock() SHARED_LOCK_FUNCTION();
  void Unlock() UNLOCK_FUNCTION();
  bool TryLock() EXCLUSIVE_TRYLOCK_FUNCTION(true);
  bool ReaderTryLock() SHARED_TRYLOCK_FUNCTION(true);

  // for negative capabilities
  const Mutex& operator!() const { return *this; }

  void AssertHeld()       ASSERT_EXCLUSIVE_LOCK();
  void AssertReaderHeld() ASSERT_SHARED_LOCK();
};


class Test {
  Mutex mu;
  int a GUARDED_BY(mu);  // expected-note3 {{guarded_by declared here}}

  void foo1() EXCLUSIVE_LOCKS_REQUIRED(mu);
  void foo2() SHARED_LOCKS_REQUIRED(mu);
  void foo3() LOCKS_EXCLUDED(mu);

  void test1() {  // expected-note {{thread warning in function 'test1'}}
    a = 0;        // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  }

  void test2() {  // expected-note {{thread warning in function 'test2'}}
    int b = a;    // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
  }

  void test3() {  // expected-note {{thread warning in function 'test3'}}
    foo1();       // expected-warning {{calling function 'foo1' requires holding mutex 'mu' exclusively}}
  }

  void test4() {  // expected-note {{thread warning in function 'test4'}}
    foo2();       // expected-warning {{calling function 'foo2' requires holding mutex 'mu'}}
  }

  void test5() {  // expected-note {{thread warning in function 'test5'}}
    mu.ReaderLock();
    foo1();       // expected-warning {{calling function 'foo1' requires holding mutex 'mu' exclusively}}
    mu.Unlock();
  }

  void test6() {  // expected-note {{thread warning in function 'test6'}}
    mu.ReaderLock();
    a = 0;        // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    mu.Unlock();
  }

  void test7() {  // expected-note {{thread warning in function 'test7'}}
    mu.Lock();
    foo3();       // expected-warning {{cannot call function 'foo3' while mutex 'mu' is held}}
    mu.Unlock();
  }
};

