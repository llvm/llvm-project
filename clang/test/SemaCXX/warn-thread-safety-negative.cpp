// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wthread-safety-negative -fcxx-exceptions -DUSE_CAPABILITY=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wthread-safety-negative -fcxx-exceptions -DUSE_CAPABILITY=1 %s

// FIXME: should also run  %clang_cc1 -fsyntax-only -verify -Wthread-safety -std=c++11 -Wc++98-compat %s
// FIXME: should also run  %clang_cc1 -fsyntax-only -verify -Wthread-safety %s

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

class LOCKABLE REENTRANT_CAPABILITY ReentrantMutex {
public:
  void Lock() EXCLUSIVE_LOCK_FUNCTION();
  void Unlock() UNLOCK_FUNCTION();

  // for negative capabilities
  const ReentrantMutex& operator!() const { return *this; }
};

class SCOPED_LOCKABLE MutexLock {
public:
  MutexLock(Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu);
  MutexLock(Mutex *mu, bool adopt) EXCLUSIVE_LOCKS_REQUIRED(mu);
  ~MutexLock() UNLOCK_FUNCTION();
};

namespace SimpleTest {

class Bar {
  Mutex mu;
  int a GUARDED_BY(mu);

public:
  void baz() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    mu.Lock();
    a = 0;
    mu.Unlock();
  }
};


class Foo {
  Mutex mu;
  int a GUARDED_BY(mu);

public:
  void foo() {
    mu.Lock();    // expected-warning {{acquiring mutex 'mu' requires negative capability '!mu'}}
    baz();        // expected-warning {{cannot call function 'baz' while mutex 'mu' is held}}
    bar();
    mu.Unlock();
  }

  void bar() {
    baz();        // expected-warning {{calling function 'baz' requires negative capability '!mu'}}
  }

  void baz() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    mu.Lock();
    a = 0;
    mu.Unlock();
  }

  void test() {
    Bar b;
    b.baz();     // no warning -- in different class.
  }

  void test2() {
    mu.Lock();   // expected-warning {{acquiring mutex 'mu' requires negative capability '!mu'}}
    a = 0;
    mu.Unlock();
    baz();       // no warning -- !mu in set.
  }

  void test3() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    mu.Lock();
    a = 0;
    mu.Unlock();
    baz();       // no warning -- !mu in set.
  }

  void test4() {
    MutexLock lock(&mu); // expected-warning {{acquiring mutex 'mu' requires negative capability '!mu'}}
  }
};

class Reentrant {
  ReentrantMutex mu;

public:
  void acquire() {
    mu.Lock();   // no warning -- reentrant mutex
    mu.Unlock();
  }

  void requireNegative() EXCLUSIVE_LOCKS_REQUIRED(!mu) { // warning?
    mu.Lock();
    mu.Unlock();
  }

  void callRequireNegative() {
    requireNegative(); // expected-warning{{calling function 'requireNegative' requires negative capability '!mu'}}
  }

  void callHaveNegative() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    requireNegative();
  }
};

class TryLockTest {
  Mutex mu;
  int a GUARDED_BY(mu);

public:
  // The analysis of a TryLock expects to have !mu declared at the boundaries,
  // even though inside a function this recursion pattern is permitted without
  // a warning.
  void tryLockNegativeWarn() {
    if (mu.TryLock()) { // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
      a = 0;
      mu.Unlock();
    }
  }

  void tryLockRebranchOneWarning(bool c) {
    bool b = mu.TryLock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    if (b)
      a = 0;
    if (c && b) {
      mu.Unlock();
    } else if (b) {
      mu.Unlock();
    }
  }

  // Inside a REQUIRES(!mu) region the declared negative fact satisfies the
  // attempt; the success edge consumes it (no duplicate '!mu' facts, no
  // spurious diagnostics), and the failure path retains it.
  void tryLockNegativeSatisfied() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    if (mu.TryLock()) {
      a = 0;
      mu.Unlock();
    } else {
      needsNegative();
    }
  }

  // Releasing an unchecked try-acquire is diagnosed at the release, and the
  // thread provably does not hold the capability afterwards: the release
  // establishes the negative fact, so re-acquiring does not also warn.
  void tryLockUncheckedReleaseThenLock() {
    mu.TryLock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    mu.Unlock();  // expected-warning{{releasing mutex 'mu' that may not be held}}
    mu.Lock();    // no '!mu' warning: the release above proves it
    a = 0;
    mu.Unlock();
  }

  void needsNegative() EXCLUSIVE_LOCKS_REQUIRED(!mu);


  // A failed try-acquire proves the negative capability on its failure
  // edge: the acquire there needs no further evidence.
  void tryLockFailureProvesNegative() {
    if (mu.TryLock()) { // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
      a = 0;
      mu.Unlock();
    } else {
      mu.Lock(); // no warning: the failed try-acquire proves '!mu'
      a = 0;
      mu.Unlock();
    }
  }
};

}  // end namespace SimpleTest

Mutex globalMutex;

namespace ScopeTest {

void f() EXCLUSIVE_LOCKS_REQUIRED(!globalMutex);
void fq() EXCLUSIVE_LOCKS_REQUIRED(!::globalMutex);

namespace ns {
  Mutex globalMutex;
  void f() EXCLUSIVE_LOCKS_REQUIRED(!globalMutex);
  void fq() EXCLUSIVE_LOCKS_REQUIRED(!ns::globalMutex);
}

void testGlobals() EXCLUSIVE_LOCKS_REQUIRED(!ns::globalMutex) {
  f();     // expected-warning {{calling function 'f' requires negative capability '!globalMutex'}}
  fq();    // expected-warning {{calling function 'fq' requires negative capability '!globalMutex'}}
  ns::f();
  ns::fq();
}

void testNamespaceGlobals() EXCLUSIVE_LOCKS_REQUIRED(!globalMutex) {
  f();
  fq();
  ns::f();  // expected-warning {{calling function 'f' requires negative capability '!globalMutex'}}
  ns::fq(); // expected-warning {{calling function 'fq' requires negative capability '!globalMutex'}}
}

class StaticMembers {
public:
  void pub() EXCLUSIVE_LOCKS_REQUIRED(!publicMutex);
  void prot() EXCLUSIVE_LOCKS_REQUIRED(!protectedMutex);
  void priv() EXCLUSIVE_LOCKS_REQUIRED(!privateMutex);
  void test() {
    pub();
    prot();
    priv();
  }

  static Mutex publicMutex;

protected:
  static Mutex protectedMutex;

private:
  static Mutex privateMutex;
};

void testStaticMembers() {
  StaticMembers x;
  x.pub();
  x.prot();
  x.priv();
}

}  // end namespace ScopeTest

namespace DoubleAttribute {

struct Foo {
  Mutex &mutex();
};

template <typename A>
class TemplateClass {
  template <typename B>
  static void Function(Foo *F)
      EXCLUSIVE_LOCKS_REQUIRED(F->mutex()) UNLOCK_FUNCTION(F->mutex()) {}
};

void test() { TemplateClass<int> TC; }

}  // end namespace DoubleAttribute
