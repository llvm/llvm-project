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
  bool TryLock() EXCLUSIVE_TRYLOCK_FUNCTION(true);

  // for negative capabilities
  const ReentrantMutex& operator!() const { return *this; }
};

class SCOPED_LOCKABLE MutexLock {
public:
  MutexLock(Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu);
  MutexLock(Mutex *mu, bool adopt) EXCLUSIVE_LOCKS_REQUIRED(mu);
  ~MutexLock() UNLOCK_FUNCTION();
};

class SCOPED_LOCKABLE MutexLockMaybe {
public:
  MutexLockMaybe(Mutex *mu) EXCLUSIVE_TRYLOCK_FUNCTION(true, mu);
  ~MutexLockMaybe() UNLOCK_FUNCTION();
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

  // A failed try-acquire of a reentrant capability does not show the
  // thread free of it: the caller may hold it.
  void tryFailureIsNotNegative() {
    if (!mu.TryLock())
      requireNegative(); // expected-warning{{calling function 'requireNegative' requires negative capability '!mu'}}
    else
      mu.Unlock();
  }

  void tryFailureKeepsDeclared() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    if (!mu.TryLock())
      requireNegative();
    else
      mu.Unlock();
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

  // Releasing an unchecked try-acquire is diagnosed at the release, and it
  // does not show the thread free of the capability (the caller may hold a
  // reentrant one): without a declared '!mu', re-acquiring needs it again.
  void tryLockUncheckedReleaseThenLock() {
    mu.TryLock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    mu.Unlock();  // expected-warning{{releasing mutex 'mu' that may not be held}}
    mu.Lock();    // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    a = 0;
    mu.Unlock();
  }

  // With '!mu' declared, the release restores it.
  void tryLockUncheckedReleaseKeepsDeclared() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    mu.TryLock();
    mu.Unlock(); // expected-warning{{releasing mutex 'mu' that may not be held}}
    mu.Lock();
    a = 0;
    mu.Unlock();
  }

  void needsNegative() EXCLUSIVE_LOCKS_REQUIRED(!mu);

  // A conditional fact carried past an unrelated join is resolved by the
  // later branch on the stored result, and the acquisition checks still run
  // only at the call (one diagnostic at the TryLock, none at the branches --
  // resolving is not a new acquisition).
  void tryLockReaddChecksOnce(bool c, bool d) {
    bool b = false;
    if (c)
      b = mu.TryLock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    if (d) {
      a = 0;            // expected-warning{{writing variable 'a' requires holding mutex 'mu' exclusively}}
    }
    if (b) {
      mu.Unlock();
    }
  }

  // A failed try-acquire proves nothing about whether the thread holds the
  // capability -- it fails for other reasons too -- so its failure edge
  // keeps what preceded the call: without a declared '!mu', the acquire
  // there needs it again.
  void tryLockFailureProvesNothing() {
    if (mu.TryLock()) { // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
      a = 0;
      mu.Unlock();
    } else {
      mu.Lock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
      a = 0;
      mu.Unlock();
    }
  }

  // With '!mu' declared, the failure edge restores what the call consumed.
  void tryLockFailureKeepsDeclared() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    if (mu.TryLock()) {
      a = 0;
      mu.Unlock();
    } else {
      mu.Lock();
      a = 0;
      mu.Unlock();
    }
  }
};

// A scoped try-acquire constructor attempts the acquisition like the
// direct call: it requires the negative capability, a declared negative
// satisfies it, and the guard's death restores only a declared one.
class ScopedTryLockTest {
  Mutex mu;
  int a GUARDED_BY(mu);

public:
  void scopedTryNegativeWarn() {
    MutexLockMaybe scope(&mu); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
  }

  // The guard's death releases whatever the try-acquire took, but does not
  // show the thread free of the capability (the caller may hold a
  // reentrant one): without a declared '!mu', a following acquire needs it
  // again.
  void scopedTryDisarmThenLock() {
    {
      MutexLockMaybe scope(&mu); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    }
    mu.Lock(); // expected-warning{{acquiring mutex 'mu' requires negative capability '!mu'}}
    a = 0;
    mu.Unlock();
  }

  // With '!mu' declared, the disarm restores it.
  void scopedTryDisarmKeepsDeclared() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    {
      MutexLockMaybe scope(&mu);
    }
    mu.Lock();
    a = 0;
    mu.Unlock();
  }

  // Inside a REQUIRES(!mu) region the declared negative fact satisfies the
  // constructor's attempt, and the disarm restores '!mu' by function end.
  void scopedTryNegativeSatisfied() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    MutexLockMaybe scope(&mu);
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
