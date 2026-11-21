// RUN: %clang_cc1 -std=c++11 %s -verify -Wthread-safety-analysis

class Mutex {
public:
  void Lock() __attribute__((exclusive_lock_function()));
  void Unlock() __attribute__((unlock_function()));
};

class A {
public:
  Mutex mu1, mu2;

  void foo() __attribute__((exclusive_locks_required(mu1))) __attribute__((exclusive_locks_required(mu2))) {}

  template <class T> void bar() __attribute__((exclusive_locks_required(mu1))) __attribute__((exclusive_locks_required(mu2))) {
    foo();
  }
};

void f() {
  A a;
  a.mu1.Lock();
  a.mu2.Lock();
  a.bar<int>();
  a.mu2.Unlock();
  a.bar<int>(); // expected-warning {{calling function 'bar<int>' requires holding mutex 'a.mu2' exclusively}}
  a.mu1.Unlock();
  a.bar<int>(); // expected-warning {{calling function 'bar<int>' requires holding mutex 'a.mu1' exclusively}} \
                   expected-warning {{calling function 'bar<int>' requires holding mutex 'a.mu2' exclusively}}
}

struct EmptyVariadicTryAcquire {
  template <typename... Mus>
  bool lockAll(Mus &...mus)
      __attribute__((try_acquire_capability(true, mus...))) {
    return true;
  }
};

void testEmptyVariadicTryAcquire(EmptyVariadicTryAcquire &h) {
  if (h.lockAll()) {
  }
}

struct EmptyVariadicCapabilities {
  template <typename... Mus>
  void lockAll(Mus &...mus) __attribute__((acquire_capability(mus...)));
  template <typename... Mus>
  void lockAllShared(Mus &...mus)
      __attribute__((acquire_shared_capability(mus...)));
  template <typename... Mus>
  void unlockAll(Mus &...mus) __attribute__((release_capability(mus...)));
  template <typename... Mus>
  void assertAll(Mus &...mus) __attribute__((assert_capability(mus...)));
};

void testEmptyVariadicCapabilities(EmptyVariadicCapabilities &h) {
  h.lockAll();
  h.lockAllShared();
  h.unlockAll();
  h.assertAll();
}

namespace TryAcquireTemplateTests {

struct VariadicHelper {
  template <typename... Mus>
  bool lockAllBool(Mus &...mus)
      __attribute__((try_acquire_capability(true, mus...))) {
    return true;
  }
  template <typename... Mus>
  bool lockAllSharedBool(Mus &...mus)
      __attribute__((try_acquire_shared_capability(true, mus...))) {
    return true;
  }
};

struct __attribute__((capability("mutex"))) VariadicCapMutex {
  void Unlock() __attribute__((release_capability()));
  template <typename... Mus>
  void LockAll(Mus &...mus) __attribute__((acquire_capability(mus...)));
  template <typename... Mus>
  bool TryAll(Mus &...mus)
      __attribute__((try_acquire_capability(true, mus...)));
};

void testEmptyVariadicPack(VariadicHelper &h, VariadicCapMutex &mu) {
  if (h.lockAllBool()) {
  }
  if (h.lockAllSharedBool()) {
  }
  // The empty packs name no capabilities, also on members of a capability
  // class, rather than the implicit object argument.
  mu.LockAll();
  mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
  if (mu.TryAll())
    mu.Unlock(); // expected-warning {{releasing mutex 'mu' that was not held}}
}

struct __attribute__((scoped_lockable)) ScopedBase {
  ScopedBase();
  ~ScopedBase() __attribute__((release_capability()));
};

// The analysis does not treat a class derived from a scoped_lockable class as
// scoped_lockable. An empty pack on its constructor acquires nothing.
struct DerivedFromScoped : ScopedBase {
  template <typename... Mus>
  DerivedFromScoped(Mus &...mus) __attribute__((acquire_capability(mus...)));
};

// An empty pack on a constructor of a class that is both scoped_lockable and a
// capability does not acquire the constructed object.
struct __attribute__((scoped_lockable, capability("mutex"))) ScopedCapability {
  template <typename... Mus>
  ScopedCapability(Mus &...mus) __attribute__((acquire_capability(mus...)));
  ~ScopedCapability() __attribute__((release_capability()));
};

// Passing a capability that is not a scoped object to a parameter with a
// release_capability attribute releases the named capability without comparing
// it with the underlying capabilities of a scoped object.
struct __attribute__((scoped_lockable, capability("mutex"))) ScopedSelfCapability {
  void LockSelf() __attribute__((acquire_capability()));
};

struct DerivedFromScopedLocking : ScopedBase {
  DerivedFromScopedLocking() __attribute__((acquire_capability()));
};

VariadicCapMutex scopedParamMu;

template <typename T>
void unlockScoped(T &s __attribute__((release_capability(scopedParamMu))));

void testReleaseNonScopedParam(ScopedSelfCapability &s) {
  s.LockSelf(); // expected-note {{mutex acquired here}}
  unlockScoped(s); // expected-warning {{releasing mutex 'scopedParamMu' that was not held}}
} // expected-warning {{mutex 's' is still held at the end of function}}

void testReleaseDerivedFromScopedParam() {
  DerivedFromScopedLocking d; // expected-note {{mutex acquired here}}
  unlockScoped(d); // expected-warning {{releasing mutex 'scopedParamMu' that was not held}}
} // expected-warning {{mutex 'd' is still held at the end of function}}

void testEmptyVariadicPackConstructors(VariadicCapMutex &mu) {
  DerivedFromScoped d;
  ScopedCapability s1;
  ScopedCapability s2(mu);
}

// Empty packs on the defer_lock and adopt_lock constructors of a variadic
// scoped lock keep the attributes, which make the analysis track the scoped
// object that its destructor releases.
struct DeferLockT {};
struct AdoptLockT {};
template <typename... Mus>
struct __attribute__((scoped_lockable)) VariadicDeferAdoptLock {
  VariadicDeferAdoptLock(Mus &...mus) __attribute__((acquire_capability(mus...)));
  VariadicDeferAdoptLock(DeferLockT, Mus &...mus)
      __attribute__((locks_excluded(mus...)));
  VariadicDeferAdoptLock(AdoptLockT, Mus &...mus)
      __attribute__((requires_capability(mus...)));
  ~VariadicDeferAdoptLock() __attribute__((release_capability()));
};

void testEmptyVariadicDeferAdoptLock() {
  VariadicDeferAdoptLock<> g1((DeferLockT()));
  VariadicDeferAdoptLock<> g2((AdoptLockT()));
}

} // namespace TryAcquireTemplateTests
