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
