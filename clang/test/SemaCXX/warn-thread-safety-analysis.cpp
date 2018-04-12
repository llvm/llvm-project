// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wno-thread-safety-negative -fcxx-exceptions -DUSE_ASSERT_CAPABILITY=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wthread-safety -Wthread-safety-beta -Wno-thread-safety-negative -fcxx-exceptions -DUSE_ASSERT_CAPABILITY=1 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety -Wthread-safety-beta -Wno-thread-safety-negative -fcxx-exceptions -DUSE_ASSERT_CAPABILITY=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety -Wthread-safety-beta -Wno-thread-safety-negative -fcxx-exceptions -DUSE_ASSERT_CAPABILITY=1 -DUSE_TRY_ACQUIRE_CAPABILITY %s

// FIXME: should also run  %clang_cc1 -fsyntax-only -verify -Wthread-safety -std=c++11 -Wc++98-compat %s
// FIXME: should also run  %clang_cc1 -fsyntax-only -verify -Wthread-safety %s

#define LOCKABLE             __attribute__((lockable))
#define SCOPED_LOCKABLE      __attribute__((scoped_lockable))
#define GUARDED_BY(x)        __attribute__((guarded_by(x)))
#define GUARDED_VAR          __attribute__((guarded_var))
#define PT_GUARDED_BY(x)     __attribute__((pt_guarded_by(x)))
#define PT_GUARDED_VAR       __attribute__((pt_guarded_var))
#define ACQUIRED_AFTER(...)  __attribute__((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__((shared_lock_function(__VA_ARGS__)))

#if USE_ASSERT_CAPABILITY
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__((assert_capability(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__((assert_shared_capability(__VA_ARGS__)))
#else
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__((assert_exclusive_lock(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__((assert_shared_lock(__VA_ARGS__)))
#endif

#ifdef USE_TRY_ACQUIRE_CAPABILITY
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__((try_acquire_capability(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__((try_acquire_shared_capability(__VA_ARGS__)))
#else
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__((shared_trylock_function(__VA_ARGS__)))
#endif
#define UNLOCK_FUNCTION(...)            __attribute__((unlock_function(__VA_ARGS__)))
#define EXCLUSIVE_UNLOCK_FUNCTION(...)  __attribute__((release_capability(__VA_ARGS__)))
#define SHARED_UNLOCK_FUNCTION(...)     __attribute__((release_shared_capability(__VA_ARGS__)))
#define LOCK_RETURNED(x)                __attribute__((lock_returned(x)))
#define LOCKS_EXCLUDED(...)             __attribute__((locks_excluded(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...)   __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...)      __attribute__((shared_locks_required(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS       __attribute__((no_thread_safety_analysis))


class LOCKABLE Mutex {
 public:
  void Lock() __attribute__((exclusive_lock_function));
  void ReaderLock() __attribute__((shared_lock_function));
  void Unlock() __attribute__((unlock_function));
  bool TryLock() EXCLUSIVE_TRYLOCK_FUNCTION(true);
  bool ReaderTryLock() SHARED_TRYLOCK_FUNCTION(true);
  void LockWhen(const int &cond) __attribute__((exclusive_lock_function));

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

class SCOPED_LOCKABLE ReaderMutexLock {
 public:
  ReaderMutexLock(Mutex *mu) SHARED_LOCK_FUNCTION(mu);
  ReaderMutexLock(Mutex *mu, bool adopt) SHARED_LOCKS_REQUIRED(mu);
  ~ReaderMutexLock() UNLOCK_FUNCTION();
};

class SCOPED_LOCKABLE ReleasableMutexLock {
 public:
  ReleasableMutexLock(Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu);
  ~ReleasableMutexLock() UNLOCK_FUNCTION();

  void Release() UNLOCK_FUNCTION();
};

class __attribute__((scoped_lockable)) DoubleMutexLock {
public:
  DoubleMutexLock(Mutex *mu1, Mutex *mu2)
      __attribute__((exclusive_lock_function(mu1, mu2)));
  ~DoubleMutexLock() __attribute__((unlock_function));
};

// The universal lock, written "*", allows checking to be selectively turned
// off for a particular piece of code.
void beginNoWarnOnReads()  SHARED_LOCK_FUNCTION("*");
void endNoWarnOnReads()    UNLOCK_FUNCTION("*");
void beginNoWarnOnWrites() EXCLUSIVE_LOCK_FUNCTION("*");
void endNoWarnOnWrites()   UNLOCK_FUNCTION("*");


// For testing handling of smart pointers.
template<class T>
class SmartPtr {
public:
  SmartPtr(T* p) : ptr_(p) { }
  SmartPtr(const SmartPtr<T>& p) : ptr_(p.ptr_) { }
  ~SmartPtr();

  T* get()        const { return ptr_; }
  T* operator->() const { return ptr_; }
  T& operator*()  const { return *ptr_; }
  T& operator[](int i) const { return ptr_[i]; }

private:
  T* ptr_;
};


// For testing destructor calls and cleanup.
class MyString {
public:
  MyString(const char* s);
  ~MyString();
};


// For testing operator overloading
template <class K, class T>
class MyMap {
public:
  T& operator[](const K& k);
};


// For testing handling of containers.
template <class T>
class MyContainer {
public:
  MyContainer();

  typedef T* iterator;
  typedef const T* const_iterator;

  T* begin();
  T* end();

  const T* cbegin();
  const T* cend();

  T&       operator[](int i);
  const T& operator[](int i) const;

private:
  T* ptr_;
};



Mutex sls_mu;

Mutex sls_mu2 __attribute__((acquired_after(sls_mu)));
int sls_guard_var __attribute__((guarded_var)) = 0;
int sls_guardby_var __attribute__((guarded_by(sls_mu))) = 0;

bool getBool();

class MutexWrapper {
public:
   Mutex mu;
   int x __attribute__((guarded_by(mu)));
   void MyLock() __attribute__((exclusive_lock_function(mu)));
};

MutexWrapper sls_mw;

void sls_fun_0() {
  sls_mw.mu.Lock();
  sls_mw.x = 5;
  sls_mw.mu.Unlock();
}

void sls_fun_2() {
  sls_mu.Lock();
  int x = sls_guard_var;
  sls_mu.Unlock();
}

void sls_fun_3() {
  sls_mu.Lock();
  sls_guard_var = 2;
  sls_mu.Unlock();
}

void sls_fun_4() {
  sls_mu2.Lock();
  sls_guard_var = 2;
  sls_mu2.Unlock();
}

void sls_fun_5() {
  sls_mu.Lock();
  int x = sls_guardby_var;
  sls_mu.Unlock();
}

void sls_fun_6() {
  sls_mu.Lock();
  sls_guardby_var = 2;
  sls_mu.Unlock();
}

void sls_fun_7() {
  sls_mu.Lock();
  sls_mu2.Lock();
  sls_mu2.Unlock();
  sls_mu.Unlock();
}

void sls_fun_8() {
  sls_mu.Lock();
  if (getBool())
    sls_mu.Unlock();
  else
    sls_mu.Unlock();
}

void sls_fun_9() {
  if (getBool())
    sls_mu.Lock();
  else
    sls_mu.Lock();
  sls_mu.Unlock();
}

void sls_fun_good_6() {
  if (getBool()) {
    sls_mu.Lock();
  } else {
    if (getBool()) {
      getBool(); // EMPTY
    } else {
      getBool(); // EMPTY
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock();
}

void sls_fun_good_7() {
  sls_mu.Lock();
  while (getBool()) {
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        sls_mu.Lock();
        continue;
      }
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock();
}

void sls_fun_good_8() {
  sls_mw.MyLock();
  sls_mw.mu.Unlock();
}

void sls_fun_bad_1() {
  sls_mu.Unlock(); // \
    // expected-warning{{releasing mutex 'sls_mu' that was not held}}
}

void sls_fun_bad_2() {
  sls_mu.Lock();
  sls_mu.Lock(); // \
    // expected-warning{{acquiring mutex 'sls_mu' that is already held}}
  sls_mu.Unlock();
}

void sls_fun_bad_3() {
  sls_mu.Lock(); // expected-note {{mutex acquired here}}
} // expected-warning{{mutex 'sls_mu' is still held at the end of function}}

void sls_fun_bad_4() {
  if (getBool())
    sls_mu.Lock();  // expected-note{{mutex acquired here}}
  else
    sls_mu2.Lock(); // expected-note{{mutex acquired here}}
} // expected-warning{{mutex 'sls_mu' is not held on every path through here}}  \
  // expected-warning{{mutex 'sls_mu2' is not held on every path through here}}

void sls_fun_bad_5() {
  sls_mu.Lock(); // expected-note {{mutex acquired here}}
  if (getBool())
    sls_mu.Unlock();
} // expected-warning{{mutex 'sls_mu' is not held on every path through here}}

void sls_fun_bad_6() {
  if (getBool()) {
    sls_mu.Lock(); // expected-note {{mutex acquired here}}
  } else {
    if (getBool()) {
      getBool(); // EMPTY
    } else {
      getBool(); // EMPTY
    }
  }
  sls_mu.Unlock(); // \
    expected-warning{{mutex 'sls_mu' is not held on every path through here}}\
    expected-warning{{releasing mutex 'sls_mu' that was not held}}
}

void sls_fun_bad_7() {
  sls_mu.Lock();
  while (getBool()) {
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        continue; // \
        expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}}
      }
    }
    sls_mu.Lock(); // expected-note {{mutex acquired here}}
  }
  sls_mu.Unlock();
}

void sls_fun_bad_8() {
  sls_mu.Lock(); // expected-note{{mutex acquired here}}

  do {
    sls_mu.Unlock(); // expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}}
  } while (getBool());
}

void sls_fun_bad_9() {
  do {
    sls_mu.Lock();  // \
      // expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}} \
      // expected-note{{mutex acquired here}}
  } while (getBool());
  sls_mu.Unlock();
}

void sls_fun_bad_10() {
  sls_mu.Lock();  // expected-note 2{{mutex acquired here}}
  while(getBool()) {  // expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
  }
} // expected-warning{{mutex 'sls_mu' is still held at the end of function}}

void sls_fun_bad_11() {
  while (getBool()) { // \
      expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}}
    sls_mu.Lock(); // expected-note {{mutex acquired here}}
  }
  sls_mu.Unlock(); // \
    // expected-warning{{releasing mutex 'sls_mu' that was not held}}
}

void sls_fun_bad_12() {
  sls_mu.Lock(); // expected-note {{mutex acquired here}}
  while (getBool()) {
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        break; // expected-warning{{mutex 'sls_mu' is not held on every path through here}}
      }
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock();
}

//-----------------------------------------//
// Handling lock expressions in attribute args
// -------------------------------------------//

Mutex aa_mu;

class GlobalLocker {
public:
  void globalLock() __attribute__((exclusive_lock_function(aa_mu)));
  void globalUnlock() __attribute__((unlock_function(aa_mu)));
};

GlobalLocker glock;

void aa_fun_1() {
  glock.globalLock();
  glock.globalUnlock();
}

void aa_fun_bad_1() {
  glock.globalUnlock(); // \
    // expected-warning{{releasing mutex 'aa_mu' that was not held}}
}

void aa_fun_bad_2() {
  glock.globalLock();
  glock.globalLock(); // \
    // expected-warning{{acquiring mutex 'aa_mu' that is already held}}
  glock.globalUnlock();
}

void aa_fun_bad_3() {
  glock.globalLock(); // expected-note{{mutex acquired here}}
} // expected-warning{{mutex 'aa_mu' is still held at the end of function}}

//--------------------------------------------------//
// Regression tests for unusual method names
//--------------------------------------------------//

Mutex wmu;

// Test diagnostics for other method names.
class WeirdMethods {
  // FIXME: can't currently check inside constructors and destructors.
  WeirdMethods() {
    wmu.Lock(); // EXPECTED-NOTE {{mutex acquired here}}
  } // EXPECTED-WARNING {{mutex 'wmu' is still held at the end of function}}
  ~WeirdMethods() {
    wmu.Lock(); // EXPECTED-NOTE {{mutex acquired here}}
  } // EXPECTED-WARNING {{mutex 'wmu' is still held at the end of function}}
  void operator++() {
    wmu.Lock(); // expected-note {{mutex acquired here}}
  } // expected-warning {{mutex 'wmu' is still held at the end of function}}
  operator int*() {
    wmu.Lock(); // expected-note {{mutex acquired here}}
    return 0;
  } // expected-warning {{mutex 'wmu' is still held at the end of function}}
};

//-----------------------------------------------//
// Errors for guarded by or guarded var variables
// ----------------------------------------------//

int *pgb_gvar __attribute__((pt_guarded_var));
int *pgb_var __attribute__((pt_guarded_by(sls_mu)));

class PGBFoo {
 public:
  int x;
  int *pgb_field __attribute__((guarded_by(sls_mu2)))
                 __attribute__((pt_guarded_by(sls_mu)));
  void testFoo() {
    pgb_field = &x; // \
      // expected-warning {{writing variable 'pgb_field' requires holding mutex 'sls_mu2' exclusively}}
    *pgb_field = x; // expected-warning {{reading variable 'pgb_field' requires holding mutex 'sls_mu2'}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires holding mutex 'sls_mu' exclusively}}
    x = *pgb_field; // expected-warning {{reading variable 'pgb_field' requires holding mutex 'sls_mu2'}} \
      // expected-warning {{reading the value pointed to by 'pgb_field' requires holding mutex 'sls_mu'}}
    (*pgb_field)++; // expected-warning {{reading variable 'pgb_field' requires holding mutex 'sls_mu2'}} \
      // expected-warning {{writing the value pointed to by 'pgb_field' requires holding mutex 'sls_mu' exclusively}}
  }
};

class GBFoo {
 public:
  int gb_field __attribute__((guarded_by(sls_mu)));

  void testFoo() {
    gb_field = 0; // \
      // expected-warning {{writing variable 'gb_field' requires holding mutex 'sls_mu' exclusively}}
  }

  void testNoAnal() __attribute__((no_thread_safety_analysis)) {
    gb_field = 0;
  }
};

GBFoo GlobalGBFoo __attribute__((guarded_by(sls_mu)));

void gb_fun_0() {
  sls_mu.Lock();
  int x = *pgb_var;
  sls_mu.Unlock();
}

void gb_fun_1() {
  sls_mu.Lock();
  *pgb_var = 2;
  sls_mu.Unlock();
}

void gb_fun_2() {
  int x;
  pgb_var = &x;
}

void gb_fun_3() {
  int *x = pgb_var;
}

void gb_bad_0() {
  sls_guard_var = 1; // \
    // expected-warning{{writing variable 'sls_guard_var' requires holding any mutex exclusively}}
}

void gb_bad_1() {
  int x = sls_guard_var; // \
    // expected-warning{{reading variable 'sls_guard_var' requires holding any mutex}}
}

void gb_bad_2() {
  sls_guardby_var = 1; // \
    // expected-warning {{writing variable 'sls_guardby_var' requires holding mutex 'sls_mu' exclusively}}
}

void gb_bad_3() {
  int x = sls_guardby_var; // \
    // expected-warning {{reading variable 'sls_guardby_var' requires holding mutex 'sls_mu'}}
}

void gb_bad_4() {
  *pgb_gvar = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_gvar' requires holding any mutex exclusively}}
}

void gb_bad_5() {
  int x = *pgb_gvar; // \
    // expected-warning {{reading the value pointed to by 'pgb_gvar' requires holding any mutex}}
}

void gb_bad_6() {
  *pgb_var = 1; // \
    // expected-warning {{writing the value pointed to by 'pgb_var' requires holding mutex 'sls_mu' exclusively}}
}

void gb_bad_7() {
  int x = *pgb_var; // \
    // expected-warning {{reading the value pointed to by 'pgb_var' requires holding mutex 'sls_mu'}}
}

void gb_bad_8() {
  GBFoo G;
  G.gb_field = 0; // \
    // expected-warning {{writing variable 'gb_field' requires holding mutex 'sls_mu'}}
}

void gb_bad_9() {
  sls_guard_var++; // \
    // expected-warning{{writing variable 'sls_guard_var' requires holding any mutex exclusively}}
  sls_guard_var--; // \
    // expected-warning{{writing variable 'sls_guard_var' requires holding any mutex exclusively}}
  ++sls_guard_var; // \
    // expected-warning{{writing variable 'sls_guard_var' requires holding any mutex exclusively}}
  --sls_guard_var;// \
    // expected-warning{{writing variable 'sls_guard_var' requires holding any mutex exclusively}}
}

//-----------------------------------------------//
// Warnings on variables with late parsed attributes
// ----------------------------------------------//

class LateFoo {
public:
  int a __attribute__((guarded_by(mu)));
  int b;

  void foo() __attribute__((exclusive_locks_required(mu))) { }

  void test() {
    a = 0; // \
      // expected-warning{{writing variable 'a' requires holding mutex 'mu' exclusively}}
    b = a; // \
      // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
    c = 0; // \
      // expected-warning {{writing variable 'c' requires holding mutex 'mu' exclusively}}
  }

  int c __attribute__((guarded_by(mu)));

  Mutex mu;
};

class LateBar {
 public:
  int a_ __attribute__((guarded_by(mu1_)));
  int b_;
  int *q __attribute__((pt_guarded_by(mu)));
  Mutex mu1_;
  Mutex mu;
  LateFoo Foo;
  LateFoo Foo2;
  LateFoo *FooPointer;
};

LateBar b1, *b3;

void late_0() {
  LateFoo FooA;
  LateFoo FooB;
  FooA.mu.Lock();
  FooA.a = 5;
  FooA.mu.Unlock();
}

void late_1() {
  LateBar BarA;
  BarA.FooPointer->mu.Lock();
  BarA.FooPointer->a = 2;
  BarA.FooPointer->mu.Unlock();
}

void late_bad_0() {
  LateFoo fooA;
  LateFoo fooB;
  fooA.mu.Lock();
  fooB.a = 5; // \
    // expected-warning{{writing variable 'a' requires holding mutex 'fooB.mu' exclusively}} \
    // expected-note{{found near match 'fooA.mu'}}
  fooA.mu.Unlock();
}

void late_bad_1() {
  Mutex mu;
  mu.Lock();
  b1.mu1_.Lock();
  int res = b1.a_ + b3->b_;
  b3->b_ = *b1.q; // \
    // expected-warning{{reading the value pointed to by 'q' requires holding mutex 'b1.mu'}}
  b1.mu1_.Unlock();
  b1.b_ = res;
  mu.Unlock();
}

void late_bad_2() {
  LateBar BarA;
  BarA.FooPointer->mu.Lock();
  BarA.Foo.a = 2; // \
    // expected-warning{{writing variable 'a' requires holding mutex 'BarA.Foo.mu' exclusively}} \
    // expected-note{{found near match 'BarA.FooPointer->mu'}}
  BarA.FooPointer->mu.Unlock();
}

void late_bad_3() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.FooPointer->a = 2; // \
    // expected-warning{{writing variable 'a' requires holding mutex 'BarA.FooPointer->mu' exclusively}} \
    // expected-note{{found near match 'BarA.Foo.mu'}}
  BarA.Foo.mu.Unlock();
}

void late_bad_4() {
  LateBar BarA;
  BarA.Foo.mu.Lock();
  BarA.Foo2.a = 2; // \
    // expected-warning{{writing variable 'a' requires holding mutex 'BarA.Foo2.mu' exclusively}} \
    // expected-note{{found near match 'BarA.Foo.mu'}}
  BarA.Foo.mu.Unlock();
}

//-----------------------------------------------//
// Extra warnings for shared vs. exclusive locks
// ----------------------------------------------//

void shared_fun_0() {
  sls_mu.Lock();
  do {
    sls_mu.Unlock();
    sls_mu.Lock();
  } while (getBool());
  sls_mu.Unlock();
}

void shared_fun_1() {
  sls_mu.ReaderLock(); // \
    // expected-warning {{mutex 'sls_mu' is acquired exclusively and shared in the same scope}}
  do {
    sls_mu.Unlock();
    sls_mu.Lock();  // \
      // expected-note {{the other acquisition of mutex 'sls_mu' is here}}
  } while (getBool());
  sls_mu.Unlock();
}

void shared_fun_3() {
  if (getBool())
    sls_mu.Lock();
  else
    sls_mu.Lock();
  *pgb_var = 1;
  sls_mu.Unlock();
}

void shared_fun_4() {
  if (getBool())
    sls_mu.ReaderLock();
  else
    sls_mu.ReaderLock();
  int x = sls_guardby_var;
  sls_mu.Unlock();
}

void shared_fun_8() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning {{mutex 'sls_mu' is acquired exclusively and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other acquisition of mutex 'sls_mu' is here}}
  sls_mu.Unlock();
}

void shared_bad_0() {
  sls_mu.Lock();  // \
    // expected-warning {{mutex 'sls_mu' is acquired exclusively and shared in the same scope}}
  do {
    sls_mu.Unlock();
    sls_mu.ReaderLock();  // \
      // expected-note {{the other acquisition of mutex 'sls_mu' is here}}
  } while (getBool());
  sls_mu.Unlock();
}

void shared_bad_1() {
  if (getBool())
    sls_mu.Lock(); // \
      // expected-warning {{mutex 'sls_mu' is acquired exclusively and shared in the same scope}}
  else
    sls_mu.ReaderLock(); // \
      // expected-note {{the other acquisition of mutex 'sls_mu' is here}}
  *pgb_var = 1;
  sls_mu.Unlock();
}

void shared_bad_2() {
  if (getBool())
    sls_mu.ReaderLock(); // \
      // expected-warning {{mutex 'sls_mu' is acquired exclusively and shared in the same scope}}
  else
    sls_mu.Lock(); // \
      // expected-note {{the other acquisition of mutex 'sls_mu' is here}}
  *pgb_var = 1;
  sls_mu.Unlock();
}

// FIXME: Add support for functions (not only methods)
class LRBar {
 public:
  void aa_elr_fun() __attribute__((exclusive_locks_required(aa_mu)));
  void aa_elr_fun_s() __attribute__((shared_locks_required(aa_mu)));
  void le_fun() __attribute__((locks_excluded(sls_mu)));
};

class LRFoo {
 public:
  void test() __attribute__((exclusive_locks_required(sls_mu)));
  void testShared() __attribute__((shared_locks_required(sls_mu2)));
};

void elr_fun() __attribute__((exclusive_locks_required(sls_mu)));
void elr_fun() {}

LRFoo MyLRFoo;
LRBar Bar;

void es_fun_0() {
  aa_mu.Lock();
  Bar.aa_elr_fun();
  aa_mu.Unlock();
}

void es_fun_1() {
  aa_mu.Lock();
  Bar.aa_elr_fun_s();
  aa_mu.Unlock();
}

void es_fun_2() {
  aa_mu.ReaderLock();
  Bar.aa_elr_fun_s();
  aa_mu.Unlock();
}

void es_fun_3() {
  sls_mu.Lock();
  MyLRFoo.test();
  sls_mu.Unlock();
}

void es_fun_4() {
  sls_mu2.Lock();
  MyLRFoo.testShared();
  sls_mu2.Unlock();
}

void es_fun_5() {
  sls_mu2.ReaderLock();
  MyLRFoo.testShared();
  sls_mu2.Unlock();
}

void es_fun_6() {
  Bar.le_fun();
}

void es_fun_7() {
  sls_mu.Lock();
  elr_fun();
  sls_mu.Unlock();
}

void es_fun_8() __attribute__((no_thread_safety_analysis));

void es_fun_8() {
  Bar.aa_elr_fun_s();
}

void es_fun_9() __attribute__((shared_locks_required(aa_mu)));
void es_fun_9() {
  Bar.aa_elr_fun_s();
}

void es_fun_10() __attribute__((exclusive_locks_required(aa_mu)));
void es_fun_10() {
  Bar.aa_elr_fun_s();
}

void es_bad_0() {
  Bar.aa_elr_fun(); // \
    // expected-warning {{calling function 'aa_elr_fun' requires holding mutex 'aa_mu' exclusively}}
}

void es_bad_1() {
  aa_mu.ReaderLock();
  Bar.aa_elr_fun(); // \
    // expected-warning {{calling function 'aa_elr_fun' requires holding mutex 'aa_mu' exclusively}}
  aa_mu.Unlock();
}

void es_bad_2() {
  Bar.aa_elr_fun_s(); // \
    // expected-warning {{calling function 'aa_elr_fun_s' requires holding mutex 'aa_mu'}}
}

void es_bad_3() {
  MyLRFoo.test(); // \
    // expected-warning {{calling function 'test' requires holding mutex 'sls_mu' exclusively}}
}

void es_bad_4() {
  MyLRFoo.testShared(); // \
    // expected-warning {{calling function 'testShared' requires holding mutex 'sls_mu2'}}
}

void es_bad_5() {
  sls_mu.ReaderLock();
  MyLRFoo.test(); // \
    // expected-warning {{calling function 'test' requires holding mutex 'sls_mu' exclusively}}
  sls_mu.Unlock();
}

void es_bad_6() {
  sls_mu.Lock();
  Bar.le_fun(); // \
    // expected-warning {{cannot call function 'le_fun' while mutex 'sls_mu' is held}}
  sls_mu.Unlock();
}

void es_bad_7() {
  sls_mu.ReaderLock();
  Bar.le_fun(); // \
    // expected-warning {{cannot call function 'le_fun' while mutex 'sls_mu' is held}}
  sls_mu.Unlock();
}


//-----------------------------------------------//
// Unparseable lock expressions
// ----------------------------------------------//

// FIXME -- derive new tests for unhandled expressions


//----------------------------------------------------------------------------//
// The following test cases are ported from the gcc thread safety implementation
// They are each wrapped inside a namespace with the test number of the gcc test
//
// FIXME: add all the gcc tests, once this analysis passes them.
//----------------------------------------------------------------------------//

//-----------------------------------------//
// Good testcases (no errors)
//-----------------------------------------//

namespace thread_annot_lock_20 {
class Bar {
 public:
  static int func1() EXCLUSIVE_LOCKS_REQUIRED(mu1_);
  static int b_ GUARDED_BY(mu1_);
  static Mutex mu1_;
  static int a_ GUARDED_BY(mu1_);
};

Bar b1;

int Bar::func1()
{
  int res = 5;

  if (a_ == 4)
    res = b_;
  return res;
}
} // end namespace thread_annot_lock_20

namespace thread_annot_lock_22 {
// Test various usage of GUARDED_BY and PT_GUARDED_BY annotations, especially
// uses in class definitions.
Mutex mu;

class Bar {
 public:
  int a_ GUARDED_BY(mu1_);
  int b_;
  int *q PT_GUARDED_BY(mu);
  Mutex mu1_ ACQUIRED_AFTER(mu);
};

Bar b1, *b3;
int *p GUARDED_BY(mu) PT_GUARDED_BY(mu);
int res GUARDED_BY(mu) = 5;

int func(int i)
{
  int x;
  mu.Lock();
  b1.mu1_.Lock();
  res = b1.a_ + b3->b_;
  *p = i;
  b1.a_ = res + b3->b_;
  b3->b_ = *b1.q;
  b1.mu1_.Unlock();
  b1.b_ = res;
  x = res;
  mu.Unlock();
  return x;
}
} // end namespace thread_annot_lock_22

namespace thread_annot_lock_27_modified {
// test lock annotations applied to function definitions
// Modified: applied annotations only to function declarations
Mutex mu1;
Mutex mu2 ACQUIRED_AFTER(mu1);

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu2) EXCLUSIVE_LOCKS_REQUIRED(mu1);
};

int Foo::method1(int i) {
  return i;
}


int foo(int i) EXCLUSIVE_LOCKS_REQUIRED(mu2) SHARED_LOCKS_REQUIRED(mu1);
int foo(int i) {
  return i;
}

static int bar(int i) EXCLUSIVE_LOCKS_REQUIRED(mu1);
static int bar(int i) {
  return i;
}

void main() {
  Foo a;

  mu1.Lock();
  mu2.Lock();
  a.method1(1);
  foo(2);
  mu2.Unlock();
  bar(3);
  mu1.Unlock();
}
} // end namespace thread_annot_lock_27_modified


namespace thread_annot_lock_38 {
// Test the case where a template member function is annotated with lock
// attributes in a non-template class.
class Foo {
 public:
  void func1(int y) LOCKS_EXCLUDED(mu_);
  template <typename T> void func2(T x) LOCKS_EXCLUDED(mu_);
 private:
  Mutex mu_;
};

Foo *foo;

void main()
{
  foo->func1(5);
  foo->func2(5);
}
} // end namespace thread_annot_lock_38

namespace thread_annot_lock_43 {
// Tests lock canonicalization
class Foo {
 public:
  Mutex *mu_;
};

class FooBar {
 public:
  Foo *foo_;
  int GetA() EXCLUSIVE_LOCKS_REQUIRED(foo_->mu_) { return a_; }
  int a_ GUARDED_BY(foo_->mu_);
};

FooBar *fb;

void main()
{
  int x;
  fb->foo_->mu_->Lock();
  x = fb->GetA();
  fb->foo_->mu_->Unlock();
}
} // end namespace thread_annot_lock_43

namespace thread_annot_lock_49 {
// Test the support for use of lock expression in the annotations
class Foo {
 public:
  Mutex foo_mu_;
};

class Bar {
 private:
  Foo *foo;
  Mutex bar_mu_ ACQUIRED_AFTER(foo->foo_mu_);

 public:
  void Test1() {
    foo->foo_mu_.Lock();
    bar_mu_.Lock();
    bar_mu_.Unlock();
    foo->foo_mu_.Unlock();
  }
};

void main() {
  Bar bar;
  bar.Test1();
}
} // end namespace thread_annot_lock_49

namespace thread_annot_lock_61_modified {
  // Modified to fix the compiler errors
  // Test the fix for a bug introduced by the support of pass-by-reference
  // parameters.
  struct Foo { Foo &operator<< (bool) {return *this;} };
  Foo &getFoo();
  struct Bar { Foo &func () {return getFoo();} };
  struct Bas { void operator& (Foo &) {} };
  void mumble()
  {
    Bas() & Bar().func() << "" << "";
    Bas() & Bar().func() << "";
  }
} // end namespace thread_annot_lock_61_modified


namespace thread_annot_lock_65 {
// Test the fix for a bug in the support of allowing reader locks for
// non-const, non-modifying overload functions. (We didn't handle the builtin
// properly.)
enum MyFlags {
  Zero,
  One,
  Two,
  Three,
  Four,
  Five,
  Six,
  Seven,
  Eight,
  Nine
};

inline MyFlags
operator|(MyFlags a, MyFlags b)
{
  return MyFlags(static_cast<int>(a) | static_cast<int>(b));
}

inline MyFlags&
operator|=(MyFlags& a, MyFlags b)
{
    return a = a | b;
}
} // end namespace thread_annot_lock_65

namespace thread_annot_lock_66_modified {
// Modified: Moved annotation to function defn
// Test annotations on out-of-line definitions of member functions where the
// annotations refer to locks that are also data members in the class.
Mutex mu;

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu1, mu, mu2);
  int data GUARDED_BY(mu1);
  Mutex *mu1;
  Mutex *mu2;
};

int Foo::method1(int i)
{
  return data + i;
}

void main()
{
  Foo a;

  a.mu2->Lock();
  a.mu1->Lock();
  mu.Lock();
  a.method1(1);
  mu.Unlock();
  a.mu1->Unlock();
  a.mu2->Unlock();
}
} // end namespace thread_annot_lock_66_modified

namespace thread_annot_lock_68_modified {
// Test a fix to a bug in the delayed name binding with nested template
// instantiation. We use a stack to make sure a name is not resolved to an
// inner context.
template <typename T>
class Bar {
  Mutex mu_;
};

template <typename T>
class Foo {
 public:
  void func(T x) {
    mu_.Lock();
    count_ = x;
    mu_.Unlock();
  }

 private:
  T count_ GUARDED_BY(mu_);
  Bar<T> bar_;
  Mutex mu_;
};

void main()
{
  Foo<int> *foo;
  foo->func(5);
}
} // end namespace thread_annot_lock_68_modified

namespace thread_annot_lock_30_modified {
// Test delay parsing of lock attribute arguments with nested classes.
// Modified: trylocks replaced with exclusive_lock_fun
int a = 0;

class Bar {
  struct Foo;

 public:
  void MyLock() EXCLUSIVE_LOCK_FUNCTION(mu);

  int func() {
    MyLock();
//    if (foo == 0) {
//      return 0;
//    }
    a = 5;
    mu.Unlock();
    return 1;
  }

  class FooBar {
    int x;
    int y;
  };

 private:
  Mutex mu;
};

Bar *bar;

void main()
{
  bar->func();
}
} // end namespace thread_annot_lock_30_modified

namespace thread_annot_lock_47 {
// Test the support for annotations on virtual functions.
// This is a good test case. (i.e. There should be no warning emitted by the
// compiler.)
class Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
  Mutex mu_;
};

class Child : public Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
};

void main() {
  Child *c;
  Base *b = c;

  b->mu_.Lock();
  b->func1();
  b->mu_.Unlock();
  b->func2();

  c->mu_.Lock();
  c->func1();
  c->mu_.Unlock();
  c->func2();
}
} // end namespace thread_annot_lock_47

//-----------------------------------------//
// Tests which produce errors
//-----------------------------------------//

namespace thread_annot_lock_13 {
Mutex mu1;
Mutex mu2;

int g GUARDED_BY(mu1);
int w GUARDED_BY(mu2);

class Foo {
 public:
  void bar() LOCKS_EXCLUDED(mu_, mu1);
  int foo() SHARED_LOCKS_REQUIRED(mu_) EXCLUSIVE_LOCKS_REQUIRED(mu2);

 private:
  int a_ GUARDED_BY(mu_);
 public:
  Mutex mu_ ACQUIRED_AFTER(mu1);
};

int Foo::foo()
{
  int res;
  w = 5;
  res = a_ + 5;
  return res;
}

void Foo::bar()
{
  int x;
  mu_.Lock();
  x = foo(); // expected-warning {{calling function 'foo' requires holding mutex 'mu2' exclusively}}
  a_ = x + 1;
  mu_.Unlock();
  if (x > 5) {
    mu1.Lock();
    g = 2;
    mu1.Unlock();
  }
}

void main()
{
  Foo f1, *f2;
  f1.mu_.Lock();
  f1.bar(); // expected-warning {{cannot call function 'bar' while mutex 'f1.mu_' is held}}
  mu2.Lock();
  f1.foo();
  mu2.Unlock();
  f1.mu_.Unlock();
  f2->mu_.Lock();
  f2->bar(); // expected-warning {{cannot call function 'bar' while mutex 'f2->mu_' is held}}
  f2->mu_.Unlock();
  mu2.Lock();
  w = 2;
  mu2.Unlock();
}
} // end namespace thread_annot_lock_13

namespace thread_annot_lock_18_modified {
// Modified: Trylocks removed
// Test the ability to distnguish between the same lock field of
// different objects of a class.
  class Bar {
 public:
  bool MyLock() EXCLUSIVE_LOCK_FUNCTION(mu1_);
  void MyUnlock() UNLOCK_FUNCTION(mu1_);
  int a_ GUARDED_BY(mu1_);

 private:
  Mutex mu1_;
};

Bar *b1, *b2;

void func()
{
  b1->MyLock();
  b1->a_ = 5;
  b2->a_ = 3; // \
    // expected-warning {{writing variable 'a_' requires holding mutex 'b2->mu1_' exclusively}} \
    // expected-note {{found near match 'b1->mu1_'}}
  b2->MyLock();
  b2->MyUnlock();
  b1->MyUnlock();
}
} // end namespace thread_annot_lock_18_modified

namespace thread_annot_lock_21 {
// Test various usage of GUARDED_BY and PT_GUARDED_BY annotations, especially
// uses in class definitions.
Mutex mu;

class Bar {
 public:
  int a_ GUARDED_BY(mu1_);
  int b_;
  int *q PT_GUARDED_BY(mu);
  Mutex mu1_ ACQUIRED_AFTER(mu);
};

Bar b1, *b3;
int *p GUARDED_BY(mu) PT_GUARDED_BY(mu);

int res GUARDED_BY(mu) = 5;

int func(int i)
{
  int x;
  b3->mu1_.Lock();
  res = b1.a_ + b3->b_; // expected-warning {{reading variable 'a_' requires holding mutex 'b1.mu1_'}} \
    // expected-warning {{writing variable 'res' requires holding mutex 'mu' exclusively}} \
    // expected-note {{found near match 'b3->mu1_'}}
  *p = i; // expected-warning {{reading variable 'p' requires holding mutex 'mu'}} \
    // expected-warning {{writing the value pointed to by 'p' requires holding mutex 'mu' exclusively}}
  b1.a_ = res + b3->b_; // expected-warning {{reading variable 'res' requires holding mutex 'mu'}} \
    // expected-warning {{writing variable 'a_' requires holding mutex 'b1.mu1_' exclusively}} \
    // expected-note {{found near match 'b3->mu1_'}}
  b3->b_ = *b1.q; // expected-warning {{reading the value pointed to by 'q' requires holding mutex 'mu'}}
  b3->mu1_.Unlock();
  b1.b_ = res; // expected-warning {{reading variable 'res' requires holding mutex 'mu'}}
  x = res; // expected-warning {{reading variable 'res' requires holding mutex 'mu'}}
  return x;
}
} // end namespace thread_annot_lock_21

namespace thread_annot_lock_35_modified {
// Test the analyzer's ability to distinguish the lock field of different
// objects.
class Foo {
 private:
  Mutex lock_;
  int a_ GUARDED_BY(lock_);

 public:
  void Func(Foo* child) LOCKS_EXCLUDED(lock_) {
     Foo *new_foo = new Foo;

     lock_.Lock();

     child->Func(new_foo); // There shouldn't be any warning here as the
                           // acquired lock is not in child.
     child->bar(7); // \
       // expected-warning {{calling function 'bar' requires holding mutex 'child->lock_' exclusively}} \
       // expected-note {{found near match 'lock_'}}
     child->a_ = 5; // \
       // expected-warning {{writing variable 'a_' requires holding mutex 'child->lock_' exclusively}} \
       // expected-note {{found near match 'lock_'}}
     lock_.Unlock();
  }

  void bar(int y) EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    a_ = y;
  }
};

Foo *x;

void main() {
  Foo *child = new Foo;
  x->Func(child);
}
} // end namespace thread_annot_lock_35_modified

namespace thread_annot_lock_36_modified {
// Modified to move the annotations to function defns.
// Test the analyzer's ability to distinguish the lock field of different
// objects
class Foo {
 private:
  Mutex lock_;
  int a_ GUARDED_BY(lock_);

 public:
  void Func(Foo* child) LOCKS_EXCLUDED(lock_);
  void bar(int y) EXCLUSIVE_LOCKS_REQUIRED(lock_);
};

void Foo::Func(Foo* child) {
  Foo *new_foo = new Foo;

  lock_.Lock();

  child->lock_.Lock();
  child->Func(new_foo); // expected-warning {{cannot call function 'Func' while mutex 'child->lock_' is held}}
  child->bar(7);
  child->a_ = 5;
  child->lock_.Unlock();

  lock_.Unlock();
}

void Foo::bar(int y) {
  a_ = y;
}


Foo *x;

void main() {
  Foo *child = new Foo;
  x->Func(child);
}
} // end namespace thread_annot_lock_36_modified


namespace thread_annot_lock_42 {
// Test support of multiple lock attributes of the same kind on a decl.
class Foo {
 private:
  Mutex mu1, mu2, mu3;
  int x GUARDED_BY(mu1) GUARDED_BY(mu2);
  int y GUARDED_BY(mu2);

  void f2() LOCKS_EXCLUDED(mu1) LOCKS_EXCLUDED(mu2) LOCKS_EXCLUDED(mu3) {
    mu2.Lock();
    y = 2;
    mu2.Unlock();
  }

 public:
  void f1() EXCLUSIVE_LOCKS_REQUIRED(mu2) EXCLUSIVE_LOCKS_REQUIRED(mu1) {
    x = 5;
    f2(); // expected-warning {{cannot call function 'f2' while mutex 'mu1' is held}} \
      // expected-warning {{cannot call function 'f2' while mutex 'mu2' is held}}
  }
};

Foo *foo;

void func()
{
  foo->f1(); // expected-warning {{calling function 'f1' requires holding mutex 'foo->mu2' exclusively}} \
             // expected-warning {{calling function 'f1' requires holding mutex 'foo->mu1' exclusively}}
}
} // end namespace thread_annot_lock_42

namespace thread_annot_lock_46 {
// Test the support for annotations on virtual functions.
class Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
  Mutex mu_;
};

class Child : public Base {
 public:
  virtual void func1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  virtual void func2() LOCKS_EXCLUDED(mu_);
};

void main() {
  Child *c;
  Base *b = c;

  b->func1(); // expected-warning {{calling function 'func1' requires holding mutex 'b->mu_' exclusively}}
  b->mu_.Lock();
  b->func2(); // expected-warning {{cannot call function 'func2' while mutex 'b->mu_' is held}}
  b->mu_.Unlock();

  c->func1(); // expected-warning {{calling function 'func1' requires holding mutex 'c->mu_' exclusively}}
  c->mu_.Lock();
  c->func2(); // expected-warning {{cannot call function 'func2' while mutex 'c->mu_' is held}}
  c->mu_.Unlock();
}
} // end namespace thread_annot_lock_46

namespace thread_annot_lock_67_modified {
// Modified: attributes on definitions moved to declarations
// Test annotations on out-of-line definitions of member functions where the
// annotations refer to locks that are also data members in the class.
Mutex mu;
Mutex mu3;

class Foo {
 public:
  int method1(int i) SHARED_LOCKS_REQUIRED(mu1, mu, mu2, mu3);
  int data GUARDED_BY(mu1);
  Mutex *mu1;
  Mutex *mu2;
};

int Foo::method1(int i) {
  return data + i;
}

void main()
{
  Foo a;
  a.method1(1); // expected-warning {{calling function 'method1' requires holding mutex 'a.mu1'}} \
    // expected-warning {{calling function 'method1' requires holding mutex 'mu'}} \
    // expected-warning {{calling function 'method1' requires holding mutex 'a.mu2'}} \
    // expected-warning {{calling function 'method1' requires holding mutex 'mu3'}}
}
} // end namespace thread_annot_lock_67_modified


namespace substitution_test {
  class MyData  {
  public:
    Mutex mu;

    void lockData()    __attribute__((exclusive_lock_function(mu)));
    void unlockData()  __attribute__((unlock_function(mu)));

    void doSomething() __attribute__((exclusive_locks_required(mu)))  { }
  };


  class DataLocker {
  public:
    void lockData  (MyData *d) __attribute__((exclusive_lock_function(d->mu)));
    void unlockData(MyData *d) __attribute__((unlock_function(d->mu)));
  };


  class Foo {
  public:
    void foo(MyData* d) __attribute__((exclusive_locks_required(d->mu))) { }

    void bar1(MyData* d) {
      d->lockData();
      foo(d);
      d->unlockData();
    }

    void bar2(MyData* d) {
      DataLocker dlr;
      dlr.lockData(d);
      foo(d);
      dlr.unlockData(d);
    }

    void bar3(MyData* d1, MyData* d2) {
      DataLocker dlr;
      dlr.lockData(d1);   // expected-note {{mutex acquired here}}
      dlr.unlockData(d2); // \
        // expected-warning {{releasing mutex 'd2->mu' that was not held}}
    } // expected-warning {{mutex 'd1->mu' is still held at the end of function}}

    void bar4(MyData* d1, MyData* d2) {
      DataLocker dlr;
      dlr.lockData(d1);
      foo(d2); // \
        // expected-warning {{calling function 'foo' requires holding mutex 'd2->mu' exclusively}} \
        // expected-note {{found near match 'd1->mu'}}
      dlr.unlockData(d1);
    }
  };
} // end namespace substituation_test



namespace constructor_destructor_tests {
  Mutex fooMu;
  int myVar GUARDED_BY(fooMu);

  class Foo {
  public:
    Foo()  __attribute__((exclusive_lock_function(fooMu))) { }
    ~Foo() __attribute__((unlock_function(fooMu))) { }
  };

  void fooTest() {
    Foo foo;
    myVar = 0;
  }
}


namespace template_member_test {

  struct S { int n; };
  struct T {
    Mutex m;
    S *s GUARDED_BY(this->m);
  };
  Mutex m;
  struct U {
    union {
      int n;
    };
  } *u GUARDED_BY(m);

  template<typename U>
  struct IndirectLock {
    int DoNaughtyThings(T *t) {
      u->n = 0; // expected-warning {{reading variable 'u' requires holding mutex 'm'}}
      return t->s->n; // expected-warning {{reading variable 's' requires holding mutex 't->m'}}
    }
  };

  template struct IndirectLock<int>; // expected-note {{here}}

  struct V {
    void f(int);
    void f(double);

    Mutex m;
    V *p GUARDED_BY(this->m);
  };
  template<typename U> struct W {
    V v;
    void f(U u) {
      v.p->f(u); // expected-warning {{reading variable 'p' requires holding mutex 'v.m'}}
    }
  };
  template struct W<int>; // expected-note {{here}}

}

namespace test_scoped_lockable {

struct TestScopedLockable {
  Mutex mu1;
  Mutex mu2;
  int a __attribute__((guarded_by(mu1)));
  int b __attribute__((guarded_by(mu2)));

  bool getBool();

  void foo1() {
    MutexLock mulock(&mu1);
    a = 5;
  }

  void foo2() {
    ReaderMutexLock mulock1(&mu1);
    if (getBool()) {
      MutexLock mulock2a(&mu2);
      b = a + 1;
    }
    else {
      MutexLock mulock2b(&mu2);
      b = a + 2;
    }
  }

  void foo3() {
    MutexLock mulock_a(&mu1);
    MutexLock mulock_b(&mu1); // \
      // expected-warning {{acquiring mutex 'mu1' that is already held}}
  }

  void foo4() {
    MutexLock mulock1(&mu1), mulock2(&mu2);
    a = b+1;
    b = a+1;
  }

  void foo5() {
    DoubleMutexLock mulock(&mu1, &mu2);
    a = b + 1;
    b = a + 1;
  }
};

} // end namespace test_scoped_lockable


namespace FunctionAttrTest {

class Foo {
public:
  Mutex mu_;
  int a GUARDED_BY(mu_);
};

Foo fooObj;

void foo() EXCLUSIVE_LOCKS_REQUIRED(fooObj.mu_);

void bar() {
  foo();  // expected-warning {{calling function 'foo' requires holding mutex 'fooObj.mu_' exclusively}}
  fooObj.mu_.Lock();
  foo();
  fooObj.mu_.Unlock();
}

};  // end namespace FunctionAttrTest


namespace TryLockTest {

struct TestTryLock {
  Mutex mu;
  int a GUARDED_BY(mu);
  bool cond;

  void foo1() {
    if (mu.TryLock()) {
      a = 1;
      mu.Unlock();
    }
  }

  void foo2() {
    if (!mu.TryLock()) return;
    a = 2;
    mu.Unlock();
  }

  void foo3() {
    bool b = mu.TryLock();
    if (b) {
      a = 3;
      mu.Unlock();
    }
  }

  void foo4() {
    bool b = mu.TryLock();
    if (!b) return;
    a = 4;
    mu.Unlock();
  }

  void foo5() {
    while (mu.TryLock()) {
      a = a + 1;
      mu.Unlock();
    }
  }

  void foo6() {
    bool b = mu.TryLock();
    b = !b;
    if (b) return;
    a = 6;
    mu.Unlock();
  }

  void foo7() {
    bool b1 = mu.TryLock();
    bool b2 = !b1;
    bool b3 = !b2;
    if (b3) {
      a = 7;
      mu.Unlock();
    }
  }

  // Test use-def chains: join points
  void foo8() {
    bool b  = mu.TryLock();
    bool b2 = b;
    if (cond)
      b = true;
    if (b) {    // b should be unknown at this point, because of the join point
      a = 8;    // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    }
    if (b2) {   // b2 should be known at this point.
      a = 8;
      mu.Unlock();
    }
  }

  // Test use-def-chains: back edges
  void foo9() {
    bool b = mu.TryLock();

    for (int i = 0; i < 10; ++i);

    if (b) {  // b is still known, because the loop doesn't alter it
      a = 9;
      mu.Unlock();
    }
  }

  // Test use-def chains: back edges
  void foo10() {
    bool b = mu.TryLock();

    while (cond) {
      if (b) {   // b should be unknown at this point b/c of the loop
        a = 10;  // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
      }
      b = !b;
    }
  }

  // Test merge of exclusive trylock
  void foo11() {
   if (cond) {
     if (!mu.TryLock())
       return;
   }
   else {
     mu.Lock();
   }
   a = 10;
   mu.Unlock();
  }

  // Test merge of shared trylock
  void foo12() {
   if (cond) {
     if (!mu.ReaderTryLock())
       return;
   }
   else {
     mu.ReaderLock();
   }
   int i = a;
   mu.Unlock();
  }
};  // end TestTrylock

} // end namespace TrylockTest


namespace TestTemplateAttributeInstantiation {

class Foo1 {
public:
  Mutex mu_;
  int a GUARDED_BY(mu_);
};

class Foo2 {
public:
  int a GUARDED_BY(mu_);
  Mutex mu_;
};


class Bar {
public:
  // Test non-dependent expressions in attributes on template functions
  template <class T>
  void barND(Foo1 *foo, T *fooT) EXCLUSIVE_LOCKS_REQUIRED(foo->mu_) {
    foo->a = 0;
  }

  // Test dependent expressions in attributes on template functions
  template <class T>
  void barD(Foo1 *foo, T *fooT) EXCLUSIVE_LOCKS_REQUIRED(fooT->mu_) {
    fooT->a = 0;
  }
};


template <class T>
class BarT {
public:
  Foo1 fooBase;
  T    fooBaseT;

  // Test non-dependent expression in ordinary method on template class
  void barND() EXCLUSIVE_LOCKS_REQUIRED(fooBase.mu_) {
    fooBase.a = 0;
  }

  // Test dependent expressions in ordinary methods on template class
  void barD() EXCLUSIVE_LOCKS_REQUIRED(fooBaseT.mu_) {
    fooBaseT.a = 0;
  }

  // Test dependent expressions in template method in template class
  template <class T2>
  void barTD(T2 *fooT) EXCLUSIVE_LOCKS_REQUIRED(fooBaseT.mu_, fooT->mu_) {
    fooBaseT.a = 0;
    fooT->a = 0;
  }
};

template <class T>
class Cell {
public:
  Mutex mu_;
  // Test dependent guarded_by
  T data GUARDED_BY(mu_);

  void fooEx() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    data = 0;
  }

  void foo() {
    mu_.Lock();
    data = 0;
    mu_.Unlock();
  }
};

void test() {
  Bar b;
  BarT<Foo2> bt;
  Foo1 f1;
  Foo2 f2;

  f1.mu_.Lock();
  f2.mu_.Lock();
  bt.fooBase.mu_.Lock();
  bt.fooBaseT.mu_.Lock();

  b.barND(&f1, &f2);
  b.barD(&f1, &f2);
  bt.barND();
  bt.barD();
  bt.barTD(&f2);

  f1.mu_.Unlock();
  bt.barTD(&f1);  // \
    // expected-warning {{calling function 'barTD<TestTemplateAttributeInstantiation::Foo1>' requires holding mutex 'f1.mu_' exclusively}} \
    // expected-note {{found near match 'bt.fooBase.mu_'}}

  bt.fooBase.mu_.Unlock();
  bt.fooBaseT.mu_.Unlock();
  f2.mu_.Unlock();

  Cell<int> cell;
  cell.data = 0; // \
    // expected-warning {{writing variable 'data' requires holding mutex 'cell.mu_' exclusively}}
  cell.foo();
  cell.mu_.Lock();
  cell.fooEx();
  cell.mu_.Unlock();
}


template <class T>
class CellDelayed {
public:
  // Test dependent guarded_by
  T data GUARDED_BY(mu_);
  static T static_data GUARDED_BY(static_mu_);

  void fooEx(CellDelayed<T> *other) EXCLUSIVE_LOCKS_REQUIRED(mu_, other->mu_) {
    this->data = other->data;
  }

  template <class T2>
  void fooExT(CellDelayed<T2> *otherT) EXCLUSIVE_LOCKS_REQUIRED(mu_, otherT->mu_) {
    this->data = otherT->data;
  }

  void foo() {
    mu_.Lock();
    data = 0;
    mu_.Unlock();
  }

  Mutex mu_;
  static Mutex static_mu_;
};

void testDelayed() {
  CellDelayed<int> celld;
  CellDelayed<int> celld2;
  celld.foo();
  celld.mu_.Lock();
  celld2.mu_.Lock();

  celld.fooEx(&celld2);
  celld.fooExT(&celld2);

  celld2.mu_.Unlock();
  celld.mu_.Unlock();
}

};  // end namespace TestTemplateAttributeInstantiation


namespace FunctionDeclDefTest {

class Foo {
public:
  Mutex mu_;
  int a GUARDED_BY(mu_);

  virtual void foo1(Foo *f_declared) EXCLUSIVE_LOCKS_REQUIRED(f_declared->mu_);
};

// EXCLUSIVE_LOCKS_REQUIRED should be applied, and rewritten to f_defined->mu_
void Foo::foo1(Foo *f_defined) {
  f_defined->a = 0;
};

void test() {
  Foo myfoo;
  myfoo.foo1(&myfoo);  // \
    // expected-warning {{calling function 'foo1' requires holding mutex 'myfoo.mu_' exclusively}}
  myfoo.mu_.Lock();
  myfoo.foo1(&myfoo);
  myfoo.mu_.Unlock();
}

};

namespace GoingNative {

  struct __attribute__((lockable)) mutex {
    void lock() __attribute__((exclusive_lock_function));
    void unlock() __attribute__((unlock_function));
    // ...
  };
  bool foo();
  bool bar();
  mutex m;
  void test() {
    m.lock();
    while (foo()) {
      m.unlock();
      // ...
      if (bar()) {
        // ...
        if (foo())
          continue; // expected-warning {{expecting mutex 'm' to be held at start of each loop}}
        //...
      }
      // ...
      m.lock(); // expected-note {{mutex acquired here}}
    }
    m.unlock();
  }

}



namespace FunctionDefinitionTest {

class Foo {
public:
  void foo1();
  void foo2();
  void foo3(Foo *other);

  template<class T>
  void fooT1(const T& dummy1);

  template<class T>
  void fooT2(const T& dummy2) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Mutex mu_;
  int a GUARDED_BY(mu_);
};

template<class T>
class FooT {
public:
  void foo();

  Mutex mu_;
  T a GUARDED_BY(mu_);
};


void Foo::foo1() NO_THREAD_SAFETY_ANALYSIS {
  a = 1;
}

void Foo::foo2() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  a = 2;
}

void Foo::foo3(Foo *other) EXCLUSIVE_LOCKS_REQUIRED(other->mu_) {
  other->a = 3;
}

template<class T>
void Foo::fooT1(const T& dummy1) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  a = dummy1;
}

/* TODO -- uncomment with template instantiation of attributes.
template<class T>
void Foo::fooT2(const T& dummy2) {
  a = dummy2;
}
*/

void fooF1(Foo *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_) {
  f->a = 1;
}

void fooF2(Foo *f);
void fooF2(Foo *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_) {
  f->a = 2;
}

void fooF3(Foo *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_);
void fooF3(Foo *f) {
  f->a = 3;
}

template<class T>
void FooT<T>::foo() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  a = 0;
}

void test() {
  int dummy = 0;
  Foo myFoo;

  myFoo.foo2();        // \
    // expected-warning {{calling function 'foo2' requires holding mutex 'myFoo.mu_' exclusively}}
  myFoo.foo3(&myFoo);  // \
    // expected-warning {{calling function 'foo3' requires holding mutex 'myFoo.mu_' exclusively}}
  myFoo.fooT1(dummy);  // \
    // expected-warning {{calling function 'fooT1<int>' requires holding mutex 'myFoo.mu_' exclusively}}

  myFoo.fooT2(dummy);  // \
    // expected-warning {{calling function 'fooT2<int>' requires holding mutex 'myFoo.mu_' exclusively}}

  fooF1(&myFoo);  // \
    // expected-warning {{calling function 'fooF1' requires holding mutex 'myFoo.mu_' exclusively}}
  fooF2(&myFoo);  // \
    // expected-warning {{calling function 'fooF2' requires holding mutex 'myFoo.mu_' exclusively}}
  fooF3(&myFoo);  // \
    // expected-warning {{calling function 'fooF3' requires holding mutex 'myFoo.mu_' exclusively}}

  myFoo.mu_.Lock();
  myFoo.foo2();
  myFoo.foo3(&myFoo);
  myFoo.fooT1(dummy);

  myFoo.fooT2(dummy);

  fooF1(&myFoo);
  fooF2(&myFoo);
  fooF3(&myFoo);
  myFoo.mu_.Unlock();

  FooT<int> myFooT;
  myFooT.foo();  // \
    // expected-warning {{calling function 'foo' requires holding mutex 'myFooT.mu_' exclusively}}
}

} // end namespace FunctionDefinitionTest


namespace SelfLockingTest {

class LOCKABLE MyLock {
public:
  int foo GUARDED_BY(this);

  void lock()   EXCLUSIVE_LOCK_FUNCTION();
  void unlock() UNLOCK_FUNCTION();

  void doSomething() {
    this->lock();  // allow 'this' as a lock expression
    foo = 0;
    doSomethingElse();
    this->unlock();
  }

  void doSomethingElse() EXCLUSIVE_LOCKS_REQUIRED(this) {
    foo = 1;
  };

  void test() {
    foo = 2;  // \
      // expected-warning {{writing variable 'foo' requires holding mutex 'this' exclusively}}
  }
};


class LOCKABLE MyLock2 {
public:
  Mutex mu_;
  int foo GUARDED_BY(this);

  // don't check inside lock and unlock functions
  void lock()   EXCLUSIVE_LOCK_FUNCTION() { mu_.Lock();   }
  void unlock() UNLOCK_FUNCTION()         { mu_.Unlock(); }

  // don't check inside constructors and destructors
  MyLock2()  { foo = 1; }
  ~MyLock2() { foo = 0; }
};


} // end namespace SelfLockingTest


namespace InvalidNonstatic {

// Forward decl here causes bogus "invalid use of non-static data member"
// on reference to mutex_ in guarded_by attribute.
class Foo;

class Foo {
  Mutex* mutex_;

  int foo __attribute__((guarded_by(mutex_)));
};

}  // end namespace InvalidNonStatic


namespace NoReturnTest {

bool condition();
void fatal() __attribute__((noreturn));

Mutex mu_;

void test1() {
  MutexLock lock(&mu_);
  if (condition()) {
    fatal();
    return;
  }
}

} // end namespace NoReturnTest


namespace TestMultiDecl {

class Foo {
public:
  int GUARDED_BY(mu_) a;
  int GUARDED_BY(mu_) b, c;

  void foo() {
    a = 0; // \
      // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    b = 0; // \
      // expected-warning {{writing variable 'b' requires holding mutex 'mu_' exclusively}}
    c = 0; // \
      // expected-warning {{writing variable 'c' requires holding mutex 'mu_' exclusively}}
  }

private:
  Mutex mu_;
};

} // end namespace TestMultiDecl


namespace WarnNoDecl {

class Foo {
  void foo(int a);  __attribute__(( // \
    // expected-warning {{declaration does not declare anything}}
    exclusive_locks_required(a))); // \
    // expected-warning {{attribute exclusive_locks_required ignored}}
};

} // end namespace WarnNoDecl



namespace MoreLockExpressions {

class Foo {
public:
  Mutex mu_;
  int a GUARDED_BY(mu_);
};

class Bar {
public:
  int b;
  Foo* f;

  Foo& getFoo()              { return *f; }
  Foo& getFoo2(int c)        { return *f; }
  Foo& getFoo3(int c, int d) { return *f; }

  Foo& getFooey() { return *f; }
};

Foo& getBarFoo(Bar &bar, int c) { return bar.getFoo2(c); }

void test() {
  Foo foo;
  Foo *fooArray;
  Bar bar;
  int a;
  int b;
  int c;

  bar.getFoo().mu_.Lock();
  bar.getFoo().a = 0;
  bar.getFoo().mu_.Unlock();

  (bar.getFoo().mu_).Lock();   // test parenthesis
  bar.getFoo().a = 0;
  (bar.getFoo().mu_).Unlock();

  bar.getFoo2(a).mu_.Lock();
  bar.getFoo2(a).a = 0;
  bar.getFoo2(a).mu_.Unlock();

  bar.getFoo3(a, b).mu_.Lock();
  bar.getFoo3(a, b).a = 0;
  bar.getFoo3(a, b).mu_.Unlock();

  getBarFoo(bar, a).mu_.Lock();
  getBarFoo(bar, a).a = 0;
  getBarFoo(bar, a).mu_.Unlock();

  bar.getFoo2(10).mu_.Lock();
  bar.getFoo2(10).a = 0;
  bar.getFoo2(10).mu_.Unlock();

  bar.getFoo2(a + 1).mu_.Lock();
  bar.getFoo2(a + 1).a = 0;
  bar.getFoo2(a + 1).mu_.Unlock();

  (a > 0 ? fooArray[1] : fooArray[b]).mu_.Lock();
  (a > 0 ? fooArray[1] : fooArray[b]).a = 0;
  (a > 0 ? fooArray[1] : fooArray[b]).mu_.Unlock();
}


void test2() {
  Foo *fooArray;
  Bar bar;
  int a;
  int b;
  int c;

  bar.getFoo().mu_.Lock();
  bar.getFooey().a = 0; // \
    // expected-warning {{writing variable 'a' requires holding mutex 'bar.getFooey().mu_' exclusively}} \
    // expected-note {{found near match 'bar.getFoo().mu_'}}
  bar.getFoo().mu_.Unlock();

  bar.getFoo2(a).mu_.Lock();
  bar.getFoo2(b).a = 0; // \
    // expected-warning {{writing variable 'a' requires holding mutex 'bar.getFoo2(b).mu_' exclusively}} \
    // expected-note {{found near match 'bar.getFoo2(a).mu_'}}
  bar.getFoo2(a).mu_.Unlock();

  bar.getFoo3(a, b).mu_.Lock();
  bar.getFoo3(a, c).a = 0;  // \
    // expected-warning {{writing variable 'a' requires holding mutex 'bar.getFoo3(a, c).mu_' exclusively}} \
    // expected-note {{found near match 'bar.getFoo3(a, b).mu_'}}
  bar.getFoo3(a, b).mu_.Unlock();

  getBarFoo(bar, a).mu_.Lock();
  getBarFoo(bar, b).a = 0;  // \
    // expected-warning {{writing variable 'a' requires holding mutex 'getBarFoo(bar, b).mu_' exclusively}} \
    // expected-note {{found near match 'getBarFoo(bar, a).mu_'}}
  getBarFoo(bar, a).mu_.Unlock();

  (a > 0 ? fooArray[1] : fooArray[b]).mu_.Lock();
  (a > 0 ? fooArray[b] : fooArray[c]).a = 0; // \
    // expected-warning {{writing variable 'a' requires holding mutex '((0 < a) ? fooArray[b] : fooArray[c]).mu_' exclusively}} \
    // expected-note {{found near match '((0 < a) ? fooArray[1] : fooArray[b]).mu_'}}
  (a > 0 ? fooArray[1] : fooArray[b]).mu_.Unlock();
}


} // end namespace MoreLockExpressions


namespace TrylockJoinPoint {

class Foo {
  Mutex mu;
  bool c;

  void foo() {
    if (c) {
      if (!mu.TryLock())
        return;
    } else {
      mu.Lock();
    }
    mu.Unlock();
  }
};

} // end namespace TrylockJoinPoint


namespace LockReturned {

class Foo {
public:
  int a             GUARDED_BY(mu_);
  void foo()        EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void foo2(Foo* f) EXCLUSIVE_LOCKS_REQUIRED(mu_, f->mu_);

  static void sfoo(Foo* f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_);

  Mutex* getMu() LOCK_RETURNED(mu_);

  Mutex mu_;

  static Mutex* getMu(Foo* f) LOCK_RETURNED(f->mu_);
};


// Calls getMu() directly to lock and unlock
void test1(Foo* f1, Foo* f2) {
  f1->a = 0;       // expected-warning {{writing variable 'a' requires holding mutex 'f1->mu_' exclusively}}
  f1->foo();       // expected-warning {{calling function 'foo' requires holding mutex 'f1->mu_' exclusively}}

  f1->foo2(f2);    // expected-warning {{calling function 'foo2' requires holding mutex 'f1->mu_' exclusively}} \
                   // expected-warning {{calling function 'foo2' requires holding mutex 'f2->mu_' exclusively}}
  Foo::sfoo(f1);   // expected-warning {{calling function 'sfoo' requires holding mutex 'f1->mu_' exclusively}}

  f1->getMu()->Lock();

  f1->a = 0;
  f1->foo();
  f1->foo2(f2); // \
    // expected-warning {{calling function 'foo2' requires holding mutex 'f2->mu_' exclusively}} \
    // expected-note {{found near match 'f1->mu_'}}

  Foo::getMu(f2)->Lock();
  f1->foo2(f2);
  Foo::getMu(f2)->Unlock();

  Foo::sfoo(f1);

  f1->getMu()->Unlock();
}


Mutex* getFooMu(Foo* f) LOCK_RETURNED(Foo::getMu(f));

class Bar : public Foo {
public:
  int  b            GUARDED_BY(getMu());
  void bar()        EXCLUSIVE_LOCKS_REQUIRED(getMu());
  void bar2(Bar* g) EXCLUSIVE_LOCKS_REQUIRED(getMu(this), g->getMu());

  static void sbar(Bar* g)  EXCLUSIVE_LOCKS_REQUIRED(g->getMu());
  static void sbar2(Bar* g) EXCLUSIVE_LOCKS_REQUIRED(getFooMu(g));
};



// Use getMu() within other attributes.
// This requires at lest levels of substitution, more in the case of
void test2(Bar* b1, Bar* b2) {
  b1->b = 0;       // expected-warning {{writing variable 'b' requires holding mutex 'b1->mu_' exclusively}}
  b1->bar();       // expected-warning {{calling function 'bar' requires holding mutex 'b1->mu_' exclusively}}
  b1->bar2(b2);    // expected-warning {{calling function 'bar2' requires holding mutex 'b1->mu_' exclusively}} \
                   // expected-warning {{calling function 'bar2' requires holding mutex 'b2->mu_' exclusively}}
  Bar::sbar(b1);   // expected-warning {{calling function 'sbar' requires holding mutex 'b1->mu_' exclusively}}
  Bar::sbar2(b1);  // expected-warning {{calling function 'sbar2' requires holding mutex 'b1->mu_' exclusively}}

  b1->getMu()->Lock();

  b1->b = 0;
  b1->bar();
  b1->bar2(b2);  // \
    // expected-warning {{calling function 'bar2' requires holding mutex 'b2->mu_' exclusively}} \
    // // expected-note {{found near match 'b1->mu_'}}

  b2->getMu()->Lock();
  b1->bar2(b2);

  b2->getMu()->Unlock();

  Bar::sbar(b1);
  Bar::sbar2(b1);

  b1->getMu()->Unlock();
}


// Sanity check -- lock the mutex directly, but use attributes that call getMu()
// Also lock the mutex using getFooMu, which calls a lock_returned function.
void test3(Bar* b1, Bar* b2) {
  b1->mu_.Lock();
  b1->b = 0;
  b1->bar();

  getFooMu(b2)->Lock();
  b1->bar2(b2);
  getFooMu(b2)->Unlock();

  Bar::sbar(b1);
  Bar::sbar2(b1);

  b1->mu_.Unlock();
}

} // end namespace LockReturned


namespace ReleasableScopedLock {

class Foo {
  Mutex mu_;
  bool c;
  int a GUARDED_BY(mu_);

  void test1();
  void test2();
  void test3();
  void test4();
  void test5();
};


void Foo::test1() {
  ReleasableMutexLock rlock(&mu_);
  rlock.Release();
}

void Foo::test2() {
  ReleasableMutexLock rlock(&mu_);
  if (c) {            // test join point -- held/not held during release
    rlock.Release();
  }
}

void Foo::test3() {
  ReleasableMutexLock rlock(&mu_);
  a = 0;
  rlock.Release();
  a = 1;  // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
}

void Foo::test4() {
  ReleasableMutexLock rlock(&mu_);
  rlock.Release();
  rlock.Release();  // expected-warning {{releasing mutex 'mu_' that was not held}}
}

void Foo::test5() {
  ReleasableMutexLock rlock(&mu_);
  if (c) {
    rlock.Release();
  }
  // no warning on join point for managed lock.
  rlock.Release();  // expected-warning {{releasing mutex 'mu_' that was not held}}
}


} // end namespace ReleasableScopedLock


namespace TrylockFunctionTest {

class Foo {
public:
  Mutex mu1_;
  Mutex mu2_;
  bool c;

  bool lockBoth() EXCLUSIVE_TRYLOCK_FUNCTION(true, mu1_, mu2_);
};

bool Foo::lockBoth() {
  if (!mu1_.TryLock())
    return false;

  mu2_.Lock();
  if (!c) {
    mu1_.Unlock();
    mu2_.Unlock();
    return false;
  }

  return true;
}


}  // end namespace TrylockFunctionTest



namespace DoubleLockBug {

class Foo {
public:
  Mutex mu_;
  int a GUARDED_BY(mu_);

  void foo1() EXCLUSIVE_LOCKS_REQUIRED(mu_);
  int  foo2() SHARED_LOCKS_REQUIRED(mu_);
};


void Foo::foo1() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  a = 0;
}

int Foo::foo2() SHARED_LOCKS_REQUIRED(mu_) {
  return a;
}

}



namespace UnlockBug {

class Foo {
public:
  Mutex mutex_;

  void foo1() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {  // expected-note {{mutex acquired here}}
    mutex_.Unlock();
  }  // expected-warning {{expecting mutex 'mutex_' to be held at the end of function}}


  void foo2() SHARED_LOCKS_REQUIRED(mutex_) {   // expected-note {{mutex acquired here}}
    mutex_.Unlock();
  }  // expected-warning {{expecting mutex 'mutex_' to be held at the end of function}}
};

} // end namespace UnlockBug



namespace FoolishScopedLockableBug {

class SCOPED_LOCKABLE WTF_ScopedLockable {
public:
  WTF_ScopedLockable(Mutex* mu) EXCLUSIVE_LOCK_FUNCTION(mu);

  // have to call release() manually;
  ~WTF_ScopedLockable();

  void release() UNLOCK_FUNCTION();
};


class Foo {
  Mutex mu_;
  int a GUARDED_BY(mu_);
  bool c;

  void doSomething();

  void test1() {
    WTF_ScopedLockable wtf(&mu_);
    wtf.release();
  }

  void test2() {
    WTF_ScopedLockable wtf(&mu_);  // expected-note {{mutex acquired here}}
  }  // expected-warning {{mutex 'mu_' is still held at the end of function}}

  void test3() {
    if (c) {
      WTF_ScopedLockable wtf(&mu_);
      wtf.release();
    }
  }

  void test4() {
    if (c) {
      doSomething();
    }
    else {
      WTF_ScopedLockable wtf(&mu_);
      wtf.release();
    }
  }

  void test5() {
    if (c) {
      WTF_ScopedLockable wtf(&mu_);  // expected-note {{mutex acquired here}}
    }
  } // expected-warning {{mutex 'mu_' is not held on every path through here}}

  void test6() {
    if (c) {
      doSomething();
    }
    else {
      WTF_ScopedLockable wtf(&mu_);  // expected-note {{mutex acquired here}}
    }
  } // expected-warning {{mutex 'mu_' is not held on every path through here}}
};


} // end namespace FoolishScopedLockableBug



namespace TemporaryCleanupExpr {

class Foo {
  int a GUARDED_BY(getMutexPtr().get());

  SmartPtr<Mutex> getMutexPtr();

  void test();
};


void Foo::test() {
  {
    ReaderMutexLock lock(getMutexPtr().get());
    int b = a;
  }
  int b = a;  // expected-warning {{reading variable 'a' requires holding mutex 'getMutexPtr()'}}
}

} // end namespace TemporaryCleanupExpr



namespace SmartPointerTests {

class Foo {
public:
  SmartPtr<Mutex> mu_;
  int a GUARDED_BY(mu_);
  int b GUARDED_BY(mu_.get());
  int c GUARDED_BY(*mu_);

  void Lock()   EXCLUSIVE_LOCK_FUNCTION(mu_);
  void Unlock() UNLOCK_FUNCTION(mu_);

  void test0();
  void test1();
  void test2();
  void test3();
  void test4();
  void test5();
  void test6();
  void test7();
  void test8();
};

void Foo::test0() {
  a = 0;  // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
  b = 0;  // expected-warning {{writing variable 'b' requires holding mutex 'mu_' exclusively}}
  c = 0;  // expected-warning {{writing variable 'c' requires holding mutex 'mu_' exclusively}}
}

void Foo::test1() {
  mu_->Lock();
  a = 0;
  b = 0;
  c = 0;
  mu_->Unlock();
}

void Foo::test2() {
  (*mu_).Lock();
  a = 0;
  b = 0;
  c = 0;
  (*mu_).Unlock();
}


void Foo::test3() {
  mu_.get()->Lock();
  a = 0;
  b = 0;
  c = 0;
  mu_.get()->Unlock();
}


void Foo::test4() {
  MutexLock lock(mu_.get());
  a = 0;
  b = 0;
  c = 0;
}


void Foo::test5() {
  MutexLock lock(&(*mu_));
  a = 0;
  b = 0;
  c = 0;
}


void Foo::test6() {
  Lock();
  a = 0;
  b = 0;
  c = 0;
  Unlock();
}


void Foo::test7() {
  {
    Lock();
    mu_->Unlock();
  }
  {
    mu_->Lock();
    Unlock();
  }
  {
    mu_.get()->Lock();
    mu_->Unlock();
  }
  {
    mu_->Lock();
    mu_.get()->Unlock();
  }
  {
    mu_.get()->Lock();
    (*mu_).Unlock();
  }
  {
    (*mu_).Lock();
    mu_->Unlock();
  }
}


void Foo::test8() {
  mu_->Lock();
  mu_.get()->Lock();    // expected-warning {{acquiring mutex 'mu_' that is already held}}
  (*mu_).Lock();        // expected-warning {{acquiring mutex 'mu_' that is already held}}
  mu_.get()->Unlock();
  Unlock();             // expected-warning {{releasing mutex 'mu_' that was not held}}
}


class Bar {
  SmartPtr<Foo> foo;

  void test0();
  void test1();
  void test2();
  void test3();
};


void Bar::test0() {
  foo->a = 0;         // expected-warning {{writing variable 'a' requires holding mutex 'foo->mu_' exclusively}}
  (*foo).b = 0;       // expected-warning {{writing variable 'b' requires holding mutex 'foo->mu_' exclusively}}
  foo.get()->c = 0;   // expected-warning {{writing variable 'c' requires holding mutex 'foo->mu_' exclusively}}
}


void Bar::test1() {
  foo->mu_->Lock();
  foo->a = 0;
  (*foo).b = 0;
  foo.get()->c = 0;
  foo->mu_->Unlock();
}


void Bar::test2() {
  (*foo).mu_->Lock();
  foo->a = 0;
  (*foo).b = 0;
  foo.get()->c = 0;
  foo.get()->mu_->Unlock();
}


void Bar::test3() {
  MutexLock lock(foo->mu_.get());
  foo->a = 0;
  (*foo).b = 0;
  foo.get()->c = 0;
}

}  // end namespace SmartPointerTests



namespace DuplicateAttributeTest {

class LOCKABLE Foo {
public:
  Mutex mu1_;
  Mutex mu2_;
  Mutex mu3_;
  int a GUARDED_BY(mu1_);
  int b GUARDED_BY(mu2_);
  int c GUARDED_BY(mu3_);

  void lock()   EXCLUSIVE_LOCK_FUNCTION();
  void unlock() UNLOCK_FUNCTION();

  void lock1()  EXCLUSIVE_LOCK_FUNCTION(mu1_);
  void slock1() SHARED_LOCK_FUNCTION(mu1_);
  void lock3()  EXCLUSIVE_LOCK_FUNCTION(mu1_, mu2_, mu3_);
  void locklots()
    EXCLUSIVE_LOCK_FUNCTION(mu1_)
    EXCLUSIVE_LOCK_FUNCTION(mu2_)
    EXCLUSIVE_LOCK_FUNCTION(mu1_, mu2_, mu3_);

  void unlock1() UNLOCK_FUNCTION(mu1_);
  void unlock3() UNLOCK_FUNCTION(mu1_, mu2_, mu3_);
  void unlocklots()
    UNLOCK_FUNCTION(mu1_)
    UNLOCK_FUNCTION(mu2_)
    UNLOCK_FUNCTION(mu1_, mu2_, mu3_);
};


void Foo::lock()   EXCLUSIVE_LOCK_FUNCTION() { }
void Foo::unlock() UNLOCK_FUNCTION()         { }

void Foo::lock1()  EXCLUSIVE_LOCK_FUNCTION(mu1_) {
  mu1_.Lock();
}

void Foo::slock1() SHARED_LOCK_FUNCTION(mu1_) {
  mu1_.ReaderLock();
}

void Foo::lock3()  EXCLUSIVE_LOCK_FUNCTION(mu1_, mu2_, mu3_) {
  mu1_.Lock();
  mu2_.Lock();
  mu3_.Lock();
}

void Foo::locklots()
    EXCLUSIVE_LOCK_FUNCTION(mu1_, mu2_)
    EXCLUSIVE_LOCK_FUNCTION(mu2_, mu3_) {
  mu1_.Lock();
  mu2_.Lock();
  mu3_.Lock();
}

void Foo::unlock1() UNLOCK_FUNCTION(mu1_) {
  mu1_.Unlock();
}

void Foo::unlock3() UNLOCK_FUNCTION(mu1_, mu2_, mu3_) {
  mu1_.Unlock();
  mu2_.Unlock();
  mu3_.Unlock();
}

void Foo::unlocklots()
    UNLOCK_FUNCTION(mu1_, mu2_)
    UNLOCK_FUNCTION(mu2_, mu3_) {
  mu1_.Unlock();
  mu2_.Unlock();
  mu3_.Unlock();
}


void test0() {
  Foo foo;
  foo.lock();
  foo.unlock();

  foo.lock();
  foo.lock();     // expected-warning {{acquiring mutex 'foo' that is already held}}
  foo.unlock();
  foo.unlock();   // expected-warning {{releasing mutex 'foo' that was not held}}
}


void test1() {
  Foo foo;
  foo.lock1();
  foo.a = 0;
  foo.unlock1();

  foo.lock1();
  foo.lock1();    // expected-warning {{acquiring mutex 'foo.mu1_' that is already held}}
  foo.a = 0;
  foo.unlock1();
  foo.unlock1();  // expected-warning {{releasing mutex 'foo.mu1_' that was not held}}
}


int test2() {
  Foo foo;
  foo.slock1();
  int d1 = foo.a;
  foo.unlock1();

  foo.slock1();
  foo.slock1();    // expected-warning {{acquiring mutex 'foo.mu1_' that is already held}}
  int d2 = foo.a;
  foo.unlock1();
  foo.unlock1();   // expected-warning {{releasing mutex 'foo.mu1_' that was not held}}
  return d1 + d2;
}


void test3() {
  Foo foo;
  foo.lock3();
  foo.a = 0;
  foo.b = 0;
  foo.c = 0;
  foo.unlock3();

  foo.lock3();
  foo.lock3(); // \
    // expected-warning {{acquiring mutex 'foo.mu1_' that is already held}} \
    // expected-warning {{acquiring mutex 'foo.mu2_' that is already held}} \
    // expected-warning {{acquiring mutex 'foo.mu3_' that is already held}}
  foo.a = 0;
  foo.b = 0;
  foo.c = 0;
  foo.unlock3();
  foo.unlock3(); // \
    // expected-warning {{releasing mutex 'foo.mu1_' that was not held}} \
    // expected-warning {{releasing mutex 'foo.mu2_' that was not held}} \
    // expected-warning {{releasing mutex 'foo.mu3_' that was not held}}
}


void testlots() {
  Foo foo;
  foo.locklots();
  foo.a = 0;
  foo.b = 0;
  foo.c = 0;
  foo.unlocklots();

  foo.locklots();
  foo.locklots(); // \
    // expected-warning {{acquiring mutex 'foo.mu1_' that is already held}} \
    // expected-warning {{acquiring mutex 'foo.mu2_' that is already held}} \
    // expected-warning {{acquiring mutex 'foo.mu3_' that is already held}}
  foo.a = 0;
  foo.b = 0;
  foo.c = 0;
  foo.unlocklots();
  foo.unlocklots(); // \
    // expected-warning {{releasing mutex 'foo.mu1_' that was not held}} \
    // expected-warning {{releasing mutex 'foo.mu2_' that was not held}} \
    // expected-warning {{releasing mutex 'foo.mu3_' that was not held}}
}

}  // end namespace DuplicateAttributeTest



namespace TryLockEqTest {

class Foo {
  Mutex mu_;
  int a GUARDED_BY(mu_);
  bool c;

  int    tryLockMutexI() EXCLUSIVE_TRYLOCK_FUNCTION(1, mu_);
  Mutex* tryLockMutexP() EXCLUSIVE_TRYLOCK_FUNCTION(1, mu_);
  void unlock() UNLOCK_FUNCTION(mu_);

  void test1();
  void test2();
};


void Foo::test1() {
  if (tryLockMutexP() == 0) {
    a = 0;  // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    return;
  }
  a = 0;
  unlock();

  if (tryLockMutexP() != 0) {
    a = 0;
    unlock();
  }

  if (0 != tryLockMutexP()) {
    a = 0;
    unlock();
  }

  if (!(tryLockMutexP() == 0)) {
    a = 0;
    unlock();
  }

  if (tryLockMutexI() == 0) {
    a = 0;   // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    return;
  }
  a = 0;
  unlock();

  if (0 == tryLockMutexI()) {
    a = 0;   // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    return;
  }
  a = 0;
  unlock();

  if (tryLockMutexI() == 1) {
    a = 0;
    unlock();
  }

  if (mu_.TryLock() == false) {
    a = 0;   // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    return;
  }
  a = 0;
  unlock();

  if (mu_.TryLock() == true) {
    a = 0;
    unlock();
  }
  else {
    a = 0;  // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
  }

#if __has_feature(cxx_nullptr)
  if (tryLockMutexP() == nullptr) {
    a = 0;  // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    return;
  }
  a = 0;
  unlock();
#endif
}

} // end namespace TryLockEqTest


namespace ExistentialPatternMatching {

class Graph {
public:
  Mutex mu_;
};

void LockAllGraphs()   EXCLUSIVE_LOCK_FUNCTION(&Graph::mu_);
void UnlockAllGraphs() UNLOCK_FUNCTION(&Graph::mu_);

class Node {
public:
  int a GUARDED_BY(&Graph::mu_);

  void foo()  EXCLUSIVE_LOCKS_REQUIRED(&Graph::mu_) {
    a = 0;
  }
  void foo2() LOCKS_EXCLUDED(&Graph::mu_);
};

void test() {
  Graph g1;
  Graph g2;
  Node n1;

  n1.a = 0;   // expected-warning {{writing variable 'a' requires holding mutex '&ExistentialPatternMatching::Graph::mu_' exclusively}}
  n1.foo();   // expected-warning {{calling function 'foo' requires holding mutex '&ExistentialPatternMatching::Graph::mu_' exclusively}}
  n1.foo2();

  g1.mu_.Lock();
  n1.a = 0;
  n1.foo();
  n1.foo2();  // expected-warning {{cannot call function 'foo2' while mutex '&ExistentialPatternMatching::Graph::mu_' is held}}
  g1.mu_.Unlock();

  g2.mu_.Lock();
  n1.a = 0;
  n1.foo();
  n1.foo2();  // expected-warning {{cannot call function 'foo2' while mutex '&ExistentialPatternMatching::Graph::mu_' is held}}
  g2.mu_.Unlock();

  LockAllGraphs();
  n1.a = 0;
  n1.foo();
  n1.foo2();  // expected-warning {{cannot call function 'foo2' while mutex '&ExistentialPatternMatching::Graph::mu_' is held}}
  UnlockAllGraphs();

  LockAllGraphs();
  g1.mu_.Unlock();

  LockAllGraphs();
  g2.mu_.Unlock();

  LockAllGraphs();
  g1.mu_.Lock();  // expected-warning {{acquiring mutex 'g1.mu_' that is already held}}
  g1.mu_.Unlock();
}

} // end namespace ExistentialPatternMatching


namespace StringIgnoreTest {

class Foo {
public:
  Mutex mu_;
  void lock()   EXCLUSIVE_LOCK_FUNCTION("");
  void unlock() UNLOCK_FUNCTION("");
  void goober() EXCLUSIVE_LOCKS_REQUIRED("");
  void roober() SHARED_LOCKS_REQUIRED("");
};


class Bar : public Foo {
public:
  void bar(Foo* f) {
    f->unlock();
    f->goober();
    f->roober();
    f->lock();
  };
};

} // end namespace StringIgnoreTest


namespace LockReturnedScopeFix {

class Base {
protected:
  struct Inner;
  bool c;

  const Mutex& getLock(const Inner* i);

  void lockInner  (Inner* i) EXCLUSIVE_LOCK_FUNCTION(getLock(i));
  void unlockInner(Inner* i) UNLOCK_FUNCTION(getLock(i));
  void foo(Inner* i) EXCLUSIVE_LOCKS_REQUIRED(getLock(i));

  void bar(Inner* i);
};


struct Base::Inner {
  Mutex lock_;
  void doSomething() EXCLUSIVE_LOCKS_REQUIRED(lock_);
};


const Mutex& Base::getLock(const Inner* i) LOCK_RETURNED(i->lock_) {
  return i->lock_;
}


void Base::foo(Inner* i) {
  i->doSomething();
}

void Base::bar(Inner* i) {
  if (c) {
    i->lock_.Lock();
    unlockInner(i);
  }
  else {
    lockInner(i);
    i->lock_.Unlock();
  }
}

} // end namespace LockReturnedScopeFix


namespace TrylockWithCleanups {

struct Foo {
  Mutex mu_;
  int a GUARDED_BY(mu_);
};

Foo* GetAndLockFoo(const MyString& s)
    EXCLUSIVE_TRYLOCK_FUNCTION(true, &Foo::mu_);

static void test() {
  Foo* lt = GetAndLockFoo("foo");
  if (!lt) return;
  int a = lt->a;
  lt->mu_.Unlock();
}

}  // end namespace TrylockWithCleanups


namespace UniversalLock {

class Foo {
  Mutex mu_;
  bool c;

  int a        GUARDED_BY(mu_);
  void r_foo() SHARED_LOCKS_REQUIRED(mu_);
  void w_foo() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void test1() {
    int b;

    beginNoWarnOnReads();
    b = a;
    r_foo();
    endNoWarnOnReads();

    beginNoWarnOnWrites();
    a = 0;
    w_foo();
    endNoWarnOnWrites();
  }

  // don't warn on joins with universal lock
  void test2() {
    if (c) {
      beginNoWarnOnWrites();
    }
    a = 0; // \
      // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
    endNoWarnOnWrites();  // \
      // expected-warning {{releasing mutex '*' that was not held}}
  }


  // make sure the universal lock joins properly
  void test3() {
    if (c) {
      mu_.Lock();
      beginNoWarnOnWrites();
    }
    else {
      beginNoWarnOnWrites();
      mu_.Lock();
    }
    a = 0;
    endNoWarnOnWrites();
    mu_.Unlock();
  }


  // combine universal lock with other locks
  void test4() {
    beginNoWarnOnWrites();
    mu_.Lock();
    mu_.Unlock();
    endNoWarnOnWrites();

    mu_.Lock();
    beginNoWarnOnWrites();
    endNoWarnOnWrites();
    mu_.Unlock();

    mu_.Lock();
    beginNoWarnOnWrites();
    mu_.Unlock();
    endNoWarnOnWrites();
  }
};

}  // end namespace UniversalLock


namespace TemplateLockReturned {

template<class T>
class BaseT {
public:
  virtual void baseMethod() = 0;
  Mutex* get_mutex() LOCK_RETURNED(mutex_) { return &mutex_; }

  Mutex mutex_;
  int a GUARDED_BY(mutex_);
};


class Derived : public BaseT<int> {
public:
  void baseMethod() EXCLUSIVE_LOCKS_REQUIRED(get_mutex()) {
    a = 0;
  }
};

}  // end namespace TemplateLockReturned


namespace ExprMatchingBugFix {

class Foo {
public:
  Mutex mu_;
};


class Bar {
public:
  bool c;
  Foo* foo;
  Bar(Foo* f) : foo(f) { }

  struct Nested {
    Foo* foo;
    Nested(Foo* f) : foo(f) { }

    void unlockFoo() UNLOCK_FUNCTION(&Foo::mu_);
  };

  void test();
};


void Bar::test() {
  foo->mu_.Lock();
  if (c) {
    Nested *n = new Nested(foo);
    n->unlockFoo();
  }
  else {
    foo->mu_.Unlock();
  }
}

}; // end namespace ExprMatchingBugfix


namespace ComplexNameTest {

class Foo {
public:
  static Mutex mu_;

  Foo() EXCLUSIVE_LOCKS_REQUIRED(mu_)  { }
  ~Foo() EXCLUSIVE_LOCKS_REQUIRED(mu_) { }

  int operator[](int i) EXCLUSIVE_LOCKS_REQUIRED(mu_) { return 0; }
};

class Bar {
public:
  static Mutex mu_;

  Bar()  LOCKS_EXCLUDED(mu_) { }
  ~Bar() LOCKS_EXCLUDED(mu_) { }

  int operator[](int i) LOCKS_EXCLUDED(mu_) { return 0; }
};


void test1() {
  Foo f;           // expected-warning {{calling function 'Foo' requires holding mutex 'mu_' exclusively}}
  int a = f[0];    // expected-warning {{calling function 'operator[]' requires holding mutex 'mu_' exclusively}}
}                  // expected-warning {{calling function '~Foo' requires holding mutex 'mu_' exclusively}}


void test2() {
  Bar::mu_.Lock();
  {
    Bar b;         // expected-warning {{cannot call function 'Bar' while mutex 'mu_' is held}}
    int a = b[0];  // expected-warning {{cannot call function 'operator[]' while mutex 'mu_' is held}}
  }                // expected-warning {{cannot call function '~Bar' while mutex 'mu_' is held}}
  Bar::mu_.Unlock();
}

};  // end namespace ComplexNameTest


namespace UnreachableExitTest {

class FemmeFatale {
public:
  FemmeFatale();
  ~FemmeFatale() __attribute__((noreturn));
};

void exitNow() __attribute__((noreturn));
void exitDestruct(const MyString& ms) __attribute__((noreturn));

Mutex fatalmu_;

void test1() EXCLUSIVE_LOCKS_REQUIRED(fatalmu_) {
  exitNow();
}

void test2() EXCLUSIVE_LOCKS_REQUIRED(fatalmu_) {
  FemmeFatale femme;
}

bool c;

void test3() EXCLUSIVE_LOCKS_REQUIRED(fatalmu_) {
  if (c) {
    exitNow();
  }
  else {
    FemmeFatale femme;
  }
}

void test4() EXCLUSIVE_LOCKS_REQUIRED(fatalmu_) {
  exitDestruct("foo");
}

}   // end namespace UnreachableExitTest


namespace VirtualMethodCanonicalizationTest {

class Base {
public:
  virtual Mutex* getMutex() = 0;
};

class Base2 : public Base {
public:
  Mutex* getMutex();
};

class Base3 : public Base2 {
public:
  Mutex* getMutex();
};

class Derived : public Base3 {
public:
  Mutex* getMutex();  // overrides Base::getMutex()
};

void baseFun(Base *b) EXCLUSIVE_LOCKS_REQUIRED(b->getMutex()) { }

void derivedFun(Derived *d) EXCLUSIVE_LOCKS_REQUIRED(d->getMutex()) {
  baseFun(d);
}

}  // end namespace VirtualMethodCanonicalizationTest


namespace TemplateFunctionParamRemapTest {

template <class T>
struct Cell {
  T dummy_;
  Mutex* mu_;
};

class Foo {
public:
  template <class T>
  void elr(Cell<T>* c) __attribute__((exclusive_locks_required(c->mu_)));

  void test();
};

template<class T>
void Foo::elr(Cell<T>* c1) { }

void Foo::test() {
  Cell<int> cell;
  elr(&cell); // \
    // expected-warning {{calling function 'elr<int>' requires holding mutex 'cell.mu_' exclusively}}
}


template<class T>
void globalELR(Cell<T>* c) __attribute__((exclusive_locks_required(c->mu_)));

template<class T>
void globalELR(Cell<T>* c1) { }

void globalTest() {
  Cell<int> cell;
  globalELR(&cell); // \
    // expected-warning {{calling function 'globalELR<int>' requires holding mutex 'cell.mu_' exclusively}}
}


template<class T>
void globalELR2(Cell<T>* c) __attribute__((exclusive_locks_required(c->mu_)));

// second declaration
template<class T>
void globalELR2(Cell<T>* c2);

template<class T>
void globalELR2(Cell<T>* c3) { }

// re-declaration after definition
template<class T>
void globalELR2(Cell<T>* c4);

void globalTest2() {
  Cell<int> cell;
  globalELR2(&cell); // \
    // expected-warning {{calling function 'globalELR2<int>' requires holding mutex 'cell.mu_' exclusively}}
}


template<class T>
class FooT {
public:
  void elr(Cell<T>* c) __attribute__((exclusive_locks_required(c->mu_)));
};

template<class T>
void FooT<T>::elr(Cell<T>* c1) { }

void testFooT() {
  Cell<int> cell;
  FooT<int> foo;
  foo.elr(&cell); // \
    // expected-warning {{calling function 'elr' requires holding mutex 'cell.mu_' exclusively}}
}

}  // end namespace TemplateFunctionParamRemapTest


namespace SelfConstructorTest {

class SelfLock {
public:
  SelfLock()  EXCLUSIVE_LOCK_FUNCTION(mu_);
  ~SelfLock() UNLOCK_FUNCTION(mu_);

  void foo() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Mutex mu_;
};

class LOCKABLE SelfLock2 {
public:
  SelfLock2()  EXCLUSIVE_LOCK_FUNCTION();
  ~SelfLock2() UNLOCK_FUNCTION();

  void foo() EXCLUSIVE_LOCKS_REQUIRED(this);
};


void test() {
  SelfLock s;
  s.foo();
}

void test2() {
  SelfLock2 s2;
  s2.foo();
}

}  // end namespace SelfConstructorTest


namespace MultipleAttributeTest {

class Foo {
  Mutex mu1_;
  Mutex mu2_;
  int  a GUARDED_BY(mu1_);
  int  b GUARDED_BY(mu2_);
  int  c GUARDED_BY(mu1_)    GUARDED_BY(mu2_);
  int* d PT_GUARDED_BY(mu1_) PT_GUARDED_BY(mu2_);

  void foo1()          EXCLUSIVE_LOCKS_REQUIRED(mu1_)
                       EXCLUSIVE_LOCKS_REQUIRED(mu2_);
  void foo2()          SHARED_LOCKS_REQUIRED(mu1_)
                       SHARED_LOCKS_REQUIRED(mu2_);
  void foo3()          LOCKS_EXCLUDED(mu1_)
                       LOCKS_EXCLUDED(mu2_);
  void lock()          EXCLUSIVE_LOCK_FUNCTION(mu1_)
                       EXCLUSIVE_LOCK_FUNCTION(mu2_);
  void readerlock()    SHARED_LOCK_FUNCTION(mu1_)
                       SHARED_LOCK_FUNCTION(mu2_);
  void unlock()        UNLOCK_FUNCTION(mu1_)
                       UNLOCK_FUNCTION(mu2_);
  bool trylock()       EXCLUSIVE_TRYLOCK_FUNCTION(true, mu1_)
                       EXCLUSIVE_TRYLOCK_FUNCTION(true, mu2_);
  bool readertrylock() SHARED_TRYLOCK_FUNCTION(true, mu1_)
                       SHARED_TRYLOCK_FUNCTION(true, mu2_);
  void assertBoth() ASSERT_EXCLUSIVE_LOCK(mu1_)
                    ASSERT_EXCLUSIVE_LOCK(mu2_);

  void alsoAssertBoth() ASSERT_EXCLUSIVE_LOCK(mu1_, mu2_);

  void assertShared() ASSERT_SHARED_LOCK(mu1_)
                      ASSERT_SHARED_LOCK(mu2_);

  void alsoAssertShared() ASSERT_SHARED_LOCK(mu1_, mu2_);

  void test();
  void testAssert();
  void testAssertShared();
};


void Foo::foo1() {
  a = 1;
  b = 2;
}

void Foo::foo2() {
  int result = a + b;
}

void Foo::foo3() { }
void Foo::lock() { mu1_.Lock();  mu2_.Lock(); }
void Foo::readerlock() { mu1_.ReaderLock();  mu2_.ReaderLock(); }
void Foo::unlock() { mu1_.Unlock();  mu2_.Unlock(); }
bool Foo::trylock()       { return true; }
bool Foo::readertrylock() { return true; }


void Foo::test() {
  mu1_.Lock();
  foo1();             // expected-warning {{}}
  c = 0;              // expected-warning {{}}
  *d = 0;             // expected-warning {{}}
  mu1_.Unlock();

  mu1_.ReaderLock();
  foo2();             // expected-warning {{}}
  int x = c;          // expected-warning {{}}
  int y = *d;         // expected-warning {{}}
  mu1_.Unlock();

  mu2_.Lock();
  foo3();             // expected-warning {{}}
  mu2_.Unlock();

  lock();
  a = 0;
  b = 0;
  unlock();

  readerlock();
  int z = a + b;
  unlock();

  if (trylock()) {
    a = 0;
    b = 0;
    unlock();
  }

  if (readertrylock()) {
    int zz = a + b;
    unlock();
  }
}

// Force duplication of attributes
void Foo::assertBoth() { }
void Foo::alsoAssertBoth() { }
void Foo::assertShared() { }
void Foo::alsoAssertShared() { }

void Foo::testAssert() {
  {
    assertBoth();
    a = 0;
    b = 0;
  }
  {
    alsoAssertBoth();
    a = 0;
    b = 0;
  }
}

void Foo::testAssertShared() {
  {
    assertShared();
    int zz = a + b;
  }

  {
    alsoAssertShared();
    int zz = a + b;
  }
}


}  // end namespace MultipleAttributeTest


namespace GuardedNonPrimitiveTypeTest {


class Data {
public:
  Data(int i) : dat(i) { }

  int  getValue() const { return dat; }
  void setValue(int i)  { dat = i; }

  int  operator[](int i) const { return dat; }
  int& operator[](int i)       { return dat; }

  void operator()() { }

private:
  int dat;
};


class DataCell {
public:
  DataCell(const Data& d) : dat(d) { }

private:
  Data dat;
};


void showDataCell(const DataCell& dc);


class Foo {
public:
  // method call tests
  void test() {
    data_.setValue(0);         // FIXME -- should be writing \
      // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
    int a = data_.getValue();  // \
      // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}

    datap1_->setValue(0);      // FIXME -- should be writing \
      // expected-warning {{reading variable 'datap1_' requires holding mutex 'mu_'}}
    a = datap1_->getValue();   // \
      // expected-warning {{reading variable 'datap1_' requires holding mutex 'mu_'}}

    datap2_->setValue(0);      // FIXME -- should be writing \
      // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}
    a = datap2_->getValue();   // \
      // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}

    (*datap2_).setValue(0);    // FIXME -- should be writing \
      // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}
    a = (*datap2_).getValue(); // \
      // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}

    mu_.Lock();
    data_.setValue(1);
    datap1_->setValue(1);
    datap2_->setValue(1);
    mu_.Unlock();

    mu_.ReaderLock();
    a = data_.getValue();
    datap1_->setValue(0);  // reads datap1_, writes *datap1_
    a = datap1_->getValue();
    a = datap2_->getValue();
    mu_.Unlock();
  }

  // operator tests
  void test2() {
    data_    = Data(1);   // expected-warning {{writing variable 'data_' requires holding mutex 'mu_' exclusively}}
    *datap1_ = data_;     // expected-warning {{reading variable 'datap1_' requires holding mutex 'mu_'}} \
                          // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
    *datap2_ = data_;     // expected-warning {{writing the value pointed to by 'datap2_' requires holding mutex 'mu_' exclusively}} \
                          // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
    data_ = *datap1_;     // expected-warning {{writing variable 'data_' requires holding mutex 'mu_' exclusively}} \
                          // expected-warning {{reading variable 'datap1_' requires holding mutex 'mu_'}}
    data_ = *datap2_;     // expected-warning {{writing variable 'data_' requires holding mutex 'mu_' exclusively}} \
                          // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}

    data_[0] = 0;         // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
    (*datap2_)[0] = 0;    // expected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}

    data_();              // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
  }

  // const operator tests
  void test3() const {
    Data mydat(data_);      // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}

    //FIXME
    //showDataCell(data_);    // xpected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
    //showDataCell(*datap2_); // xpected-warning {{reading the value pointed to by 'datap2_' requires holding mutex 'mu_'}}

    int a = data_[0];       // expected-warning {{reading variable 'data_' requires holding mutex 'mu_'}}
  }

private:
  Mutex mu_;
  Data  data_   GUARDED_BY(mu_);
  Data* datap1_ GUARDED_BY(mu_);
  Data* datap2_ PT_GUARDED_BY(mu_);
};

}  // end namespace GuardedNonPrimitiveTypeTest


namespace GuardedNonPrimitive_MemberAccess {

class Cell {
public:
  Cell(int i);

  void cellMethod();

  int a;
};


class Foo {
public:
  int   a;
  Cell  c  GUARDED_BY(cell_mu_);
  Cell* cp PT_GUARDED_BY(cell_mu_);

  void myMethod();

  Mutex cell_mu_;
};


class Bar {
private:
  Mutex mu_;
  Foo  foo  GUARDED_BY(mu_);
  Foo* foop PT_GUARDED_BY(mu_);

  void test() {
    foo.myMethod();      // expected-warning {{reading variable 'foo' requires holding mutex 'mu_'}}

    int fa = foo.a;      // expected-warning {{reading variable 'foo' requires holding mutex 'mu_'}}
    foo.a  = fa;         // expected-warning {{writing variable 'foo' requires holding mutex 'mu_' exclusively}}

    fa = foop->a;        // expected-warning {{reading the value pointed to by 'foop' requires holding mutex 'mu_'}}
    foop->a = fa;        // expected-warning {{writing the value pointed to by 'foop' requires holding mutex 'mu_' exclusively}}

    fa = (*foop).a;      // expected-warning {{reading the value pointed to by 'foop' requires holding mutex 'mu_'}}
    (*foop).a = fa;      // expected-warning {{writing the value pointed to by 'foop' requires holding mutex 'mu_' exclusively}}

    foo.c  = Cell(0);    // expected-warning {{writing variable 'foo' requires holding mutex 'mu_'}} \
                         // expected-warning {{writing variable 'c' requires holding mutex 'foo.cell_mu_' exclusively}}
    foo.c.cellMethod();  // expected-warning {{reading variable 'foo' requires holding mutex 'mu_'}} \
                         // expected-warning {{reading variable 'c' requires holding mutex 'foo.cell_mu_'}}

    foop->c  = Cell(0);    // expected-warning {{writing the value pointed to by 'foop' requires holding mutex 'mu_'}} \
                           // expected-warning {{writing variable 'c' requires holding mutex 'foop->cell_mu_' exclusively}}
    foop->c.cellMethod();  // expected-warning {{reading the value pointed to by 'foop' requires holding mutex 'mu_'}} \
                           // expected-warning {{reading variable 'c' requires holding mutex 'foop->cell_mu_'}}

    (*foop).c  = Cell(0);    // expected-warning {{writing the value pointed to by 'foop' requires holding mutex 'mu_'}} \
                             // expected-warning {{writing variable 'c' requires holding mutex 'foop->cell_mu_' exclusively}}
    (*foop).c.cellMethod();  // expected-warning {{reading the value pointed to by 'foop' requires holding mutex 'mu_'}} \
                             // expected-warning {{reading variable 'c' requires holding mutex 'foop->cell_mu_'}}
  };
};

}  // namespace GuardedNonPrimitive_MemberAccess


namespace TestThrowExpr {

class Foo {
  Mutex mu_;

  bool hasError();

  void test() {
    mu_.Lock();
    if (hasError()) {
      throw "ugly";
    }
    mu_.Unlock();
  }
};

}  // end namespace TestThrowExpr


namespace UnevaluatedContextTest {

// parse attribute expressions in an unevaluated context.

static inline Mutex* getMutex1();
static inline Mutex* getMutex2();

void bar() EXCLUSIVE_LOCKS_REQUIRED(getMutex1());

void bar2() EXCLUSIVE_LOCKS_REQUIRED(getMutex1(), getMutex2());

}  // end namespace UnevaluatedContextTest


namespace LockUnlockFunctionTest {

// Check built-in lock functions
class LOCKABLE MyLockable  {
public:
  void lock()       EXCLUSIVE_LOCK_FUNCTION() { mu_.Lock(); }
  void readerLock() SHARED_LOCK_FUNCTION()    { mu_.ReaderLock(); }
  void unlock()     UNLOCK_FUNCTION()         { mu_.Unlock(); }

private:
  Mutex mu_;
};


class Foo {
public:
  // Correct lock/unlock functions
  void lock() EXCLUSIVE_LOCK_FUNCTION(mu_) {
    mu_.Lock();
  }

  void readerLock() SHARED_LOCK_FUNCTION(mu_) {
    mu_.ReaderLock();
  }

  void unlock() UNLOCK_FUNCTION(mu_) {
    mu_.Unlock();
  }

  // Check failure to lock.
  void lockBad() EXCLUSIVE_LOCK_FUNCTION(mu_) {    // expected-note {{mutex acquired here}}
    mu2_.Lock();
    mu2_.Unlock();
  }  // expected-warning {{expecting mutex 'mu_' to be held at the end of function}}

  void readerLockBad() SHARED_LOCK_FUNCTION(mu_) {  // expected-note {{mutex acquired here}}
    mu2_.Lock();
    mu2_.Unlock();
  }  // expected-warning {{expecting mutex 'mu_' to be held at the end of function}}

  void unlockBad() UNLOCK_FUNCTION(mu_) {  // expected-note {{mutex acquired here}}
    mu2_.Lock();
    mu2_.Unlock();
  }  // expected-warning {{mutex 'mu_' is still held at the end of function}}

  // Check locking the wrong thing.
  void lockBad2() EXCLUSIVE_LOCK_FUNCTION(mu_) {   // expected-note {{mutex acquired here}}
    mu2_.Lock();            // expected-note {{mutex acquired here}}
  } // expected-warning {{expecting mutex 'mu_' to be held at the end of function}} \
    // expected-warning {{mutex 'mu2_' is still held at the end of function}}


  void readerLockBad2() SHARED_LOCK_FUNCTION(mu_) {   // expected-note {{mutex acquired here}}
    mu2_.ReaderLock();      // expected-note {{mutex acquired here}}
  } // expected-warning {{expecting mutex 'mu_' to be held at the end of function}} \
    // expected-warning {{mutex 'mu2_' is still held at the end of function}}


  void unlockBad2() UNLOCK_FUNCTION(mu_) {  // expected-note {{mutex acquired here}}
    mu2_.Unlock();  // expected-warning {{releasing mutex 'mu2_' that was not held}}
  }  // expected-warning {{mutex 'mu_' is still held at the end of function}}

private:
  Mutex mu_;
  Mutex mu2_;
};

}  // end namespace LockUnlockFunctionTest


namespace AssertHeldTest {

class Foo {
public:
  int c;
  int a GUARDED_BY(mu_);
  Mutex mu_;

  void test1() {
    mu_.AssertHeld();
    int b = a;
    a = 0;
  }

  void test2() {
    mu_.AssertReaderHeld();
    int b = a;
    a = 0;   // expected-warning {{writing variable 'a' requires holding mutex 'mu_' exclusively}}
  }

  void test3() {
    if (c) {
      mu_.AssertHeld();
    }
    else {
      mu_.AssertHeld();
    }
    int b = a;
    a = 0;
  }

  void test4() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    mu_.AssertHeld();
    int b = a;
    a = 0;
  }

  void test5() UNLOCK_FUNCTION(mu_) {
    mu_.AssertHeld();
    mu_.Unlock();
  }

  void test6() {
    mu_.AssertHeld();
    mu_.Unlock();
  }  // should this be a warning?

  void test7() {
    if (c) {
      mu_.AssertHeld();
    }
    else {
      mu_.Lock();
    }
    int b = a;
    a = 0;
    mu_.Unlock();
  }

  void test8() {
    if (c) {
      mu_.Lock();
    }
    else {
      mu_.AssertHeld();
    }
    int b = a;
    a = 0;
    mu_.Unlock();
  }

  void test9() {
    if (c) {
      mu_.AssertHeld();
    }
    else {
      mu_.Lock();  // expected-note {{mutex acquired here}}
    }
  }  // expected-warning {{mutex 'mu_' is still held at the end of function}}

  void test10() {
    if (c) {
      mu_.Lock();  // expected-note {{mutex acquired here}}
    }
    else {
      mu_.AssertHeld();
    }
  }  // expected-warning {{mutex 'mu_' is still held at the end of function}}

  void assertMu() ASSERT_EXCLUSIVE_LOCK(mu_);

  void test11() {
    assertMu();
    int b = a;
    a = 0;
  }
};

}  // end namespace AssertHeldTest


namespace LogicalConditionalTryLock {

class Foo {
public:
  Mutex mu;
  int a GUARDED_BY(mu);
  bool c;

  bool newc();

  void test1() {
    if (c && mu.TryLock()) {
      a = 0;
      mu.Unlock();
    }
  }

  void test2() {
    bool b = mu.TryLock();
    if (c && b) {
      a = 0;
      mu.Unlock();
    }
  }

  void test3() {
    if (c || !mu.TryLock())
      return;
    a = 0;
    mu.Unlock();
  }

  void test4() {
    while (c && mu.TryLock()) {
      a = 0;
      c = newc();
      mu.Unlock();
    }
  }

  void test5() {
    while (c) {
      if (newc() || !mu.TryLock())
        break;
      a = 0;
      mu.Unlock();
    }
  }

  void test6() {
    mu.Lock();
    do {
      a = 0;
      mu.Unlock();
    } while (newc() && mu.TryLock());
  }

  void test7() {
    for (bool b = mu.TryLock(); c && b;) {
      a = 0;
      mu.Unlock();
    }
  }

  void test8() {
    if (c && newc() && mu.TryLock()) {
      a = 0;
      mu.Unlock();
    }
  }

  void test9() {
    if (!(c && newc() && mu.TryLock()))
      return;
    a = 0;
    mu.Unlock();
  }

  void test10() {
    if (!(c || !mu.TryLock())) {
      a = 0;
      mu.Unlock();
    }
  }
};

}  // end namespace LogicalConditionalTryLock



namespace PtGuardedByTest {

void doSomething();

class Cell {
  public:
  int a;
};


// This mainly duplicates earlier tests, but just to make sure...
class PtGuardedBySanityTest {
  Mutex  mu1;
  Mutex  mu2;
  int*   a GUARDED_BY(mu1) PT_GUARDED_BY(mu2);
  Cell*  c GUARDED_BY(mu1) PT_GUARDED_BY(mu2);
  int    sa[10] GUARDED_BY(mu1);
  Cell   sc[10] GUARDED_BY(mu1);

  void test1() {
    mu1.Lock();
    if (a == 0) doSomething();  // OK, we don't dereference.
    a = 0;
    c = 0;
    if (sa[0] == 42) doSomething();
    sa[0] = 57;
    if (sc[0].a == 42) doSomething();
    sc[0].a = 57;
    mu1.Unlock();
  }

  void test2() {
    mu1.ReaderLock();
    if (*a == 0) doSomething();      // expected-warning {{reading the value pointed to by 'a' requires holding mutex 'mu2'}}
    *a = 0;                          // expected-warning {{writing the value pointed to by 'a' requires holding mutex 'mu2' exclusively}}

    if (c->a == 0) doSomething();    // expected-warning {{reading the value pointed to by 'c' requires holding mutex 'mu2'}}
    c->a = 0;                        // expected-warning {{writing the value pointed to by 'c' requires holding mutex 'mu2' exclusively}}

    if ((*c).a == 0) doSomething();  // expected-warning {{reading the value pointed to by 'c' requires holding mutex 'mu2'}}
    (*c).a = 0;                      // expected-warning {{writing the value pointed to by 'c' requires holding mutex 'mu2' exclusively}}

    if (a[0] == 42) doSomething();     // expected-warning {{reading the value pointed to by 'a' requires holding mutex 'mu2'}}
    a[0] = 57;                         // expected-warning {{writing the value pointed to by 'a' requires holding mutex 'mu2' exclusively}}
    if (c[0].a == 42) doSomething();   // expected-warning {{reading the value pointed to by 'c' requires holding mutex 'mu2'}}
    c[0].a = 57;                       // expected-warning {{writing the value pointed to by 'c' requires holding mutex 'mu2' exclusively}}
    mu1.Unlock();
  }

  void test3() {
    mu2.Lock();
    if (*a == 0) doSomething();      // expected-warning {{reading variable 'a' requires holding mutex 'mu1'}}
    *a = 0;                          // expected-warning {{reading variable 'a' requires holding mutex 'mu1'}}

    if (c->a == 0) doSomething();    // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}
    c->a = 0;                        // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}

    if ((*c).a == 0) doSomething();  // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}
    (*c).a = 0;                      // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}

    if (a[0] == 42) doSomething();     // expected-warning {{reading variable 'a' requires holding mutex 'mu1'}}
    a[0] = 57;                         // expected-warning {{reading variable 'a' requires holding mutex 'mu1'}}
    if (c[0].a == 42) doSomething();   // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}
    c[0].a = 57;                       // expected-warning {{reading variable 'c' requires holding mutex 'mu1'}}
    mu2.Unlock();
  }

  void test4() {  // Literal arrays
    if (sa[0] == 42) doSomething();     // expected-warning {{reading variable 'sa' requires holding mutex 'mu1'}}
    sa[0] = 57;                         // expected-warning {{writing variable 'sa' requires holding mutex 'mu1' exclusively}}
    if (sc[0].a == 42) doSomething();   // expected-warning {{reading variable 'sc' requires holding mutex 'mu1'}}
    sc[0].a = 57;                       // expected-warning {{writing variable 'sc' requires holding mutex 'mu1' exclusively}}

    if (*sa == 42) doSomething();       // expected-warning {{reading variable 'sa' requires holding mutex 'mu1'}}
    *sa = 57;                           // expected-warning {{writing variable 'sa' requires holding mutex 'mu1' exclusively}}
    if ((*sc).a == 42) doSomething();   // expected-warning {{reading variable 'sc' requires holding mutex 'mu1'}}
    (*sc).a = 57;                       // expected-warning {{writing variable 'sc' requires holding mutex 'mu1' exclusively}}
    if (sc->a == 42) doSomething();     // expected-warning {{reading variable 'sc' requires holding mutex 'mu1'}}
    sc->a = 57;                         // expected-warning {{writing variable 'sc' requires holding mutex 'mu1' exclusively}}
  }

  void test5() {
    mu1.ReaderLock();    // OK -- correct use.
    mu2.Lock();
    if (*a == 0) doSomething();
    *a = 0;

    if (c->a == 0) doSomething();
    c->a = 0;

    if ((*c).a == 0) doSomething();
    (*c).a = 0;
    mu2.Unlock();
    mu1.Unlock();
  }
};


class SmartPtr_PtGuardedBy_Test {
  Mutex mu1;
  Mutex mu2;
  SmartPtr<int>  sp GUARDED_BY(mu1) PT_GUARDED_BY(mu2);
  SmartPtr<Cell> sq GUARDED_BY(mu1) PT_GUARDED_BY(mu2);

  void test1() {
    mu1.ReaderLock();
    mu2.Lock();

    sp.get();
    if (*sp == 0) doSomething();
    *sp = 0;
    sq->a = 0;

    if (sp[0] == 0) doSomething();
    sp[0] = 0;

    mu2.Unlock();
    mu1.Unlock();
  }

  void test2() {
    mu2.Lock();

    sp.get();                      // expected-warning {{reading variable 'sp' requires holding mutex 'mu1'}}
    if (*sp == 0) doSomething();   // expected-warning {{reading variable 'sp' requires holding mutex 'mu1'}}
    *sp = 0;                       // expected-warning {{reading variable 'sp' requires holding mutex 'mu1'}}
    sq->a = 0;                     // expected-warning {{reading variable 'sq' requires holding mutex 'mu1'}}

    if (sp[0] == 0) doSomething();   // expected-warning {{reading variable 'sp' requires holding mutex 'mu1'}}
    sp[0] = 0;                       // expected-warning {{reading variable 'sp' requires holding mutex 'mu1'}}
    if (sq[0].a == 0) doSomething(); // expected-warning {{reading variable 'sq' requires holding mutex 'mu1'}}
    sq[0].a = 0;                     // expected-warning {{reading variable 'sq' requires holding mutex 'mu1'}}

    mu2.Unlock();
  }

  void test3() {
    mu1.Lock();

    sp.get();
    if (*sp == 0) doSomething();   // expected-warning {{reading the value pointed to by 'sp' requires holding mutex 'mu2'}}
    *sp = 0;                       // expected-warning {{reading the value pointed to by 'sp' requires holding mutex 'mu2'}}
    sq->a = 0;                     // expected-warning {{reading the value pointed to by 'sq' requires holding mutex 'mu2'}}

    if (sp[0] == 0) doSomething();   // expected-warning {{reading the value pointed to by 'sp' requires holding mutex 'mu2'}}
    sp[0] = 0;                       // expected-warning {{reading the value pointed to by 'sp' requires holding mutex 'mu2'}}
    if (sq[0].a == 0) doSomething(); // expected-warning {{reading the value pointed to by 'sq' requires holding mutex 'mu2'}}
    sq[0].a = 0;                     // expected-warning {{reading the value pointed to by 'sq' requires holding mutex 'mu2'}}

    mu1.Unlock();
  }
};

}  // end namespace PtGuardedByTest


namespace NonMemberCalleeICETest {

class A {
  void Run() {
  (RunHelper)();  // expected-warning {{calling function 'RunHelper' requires holding mutex 'M' exclusively}}
 }

 void RunHelper() __attribute__((exclusive_locks_required(M)));
 Mutex M;
};

}  // end namespace NonMemberCalleeICETest


namespace pt_guard_attribute_type {
  int i PT_GUARDED_BY(sls_mu);  // expected-warning {{'pt_guarded_by' only applies to pointer types; type here is 'int'}}
  int j PT_GUARDED_VAR;  // expected-warning {{'pt_guarded_var' only applies to pointer types; type here is 'int'}}

  void test() {
    int i PT_GUARDED_BY(sls_mu);  // expected-warning {{'pt_guarded_by' attribute only applies to non-static data members and global variables}}
    int j PT_GUARDED_VAR;  // expected-warning {{'pt_guarded_var' attribute only applies to non-static data members and global variables}}

    typedef int PT_GUARDED_BY(sls_mu) bad1;  // expected-warning {{'pt_guarded_by' attribute only applies to}}
    typedef int PT_GUARDED_VAR bad2;  // expected-warning {{'pt_guarded_var' attribute only applies to}}
  }
}  // end namespace pt_guard_attribute_type


namespace ThreadAttributesOnLambdas {

class Foo {
  Mutex mu_;

  void LockedFunction() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void test() {
    auto func1 = [this]() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      LockedFunction();
    };

    auto func2 = [this]() NO_THREAD_SAFETY_ANALYSIS {
      LockedFunction();
    };

    auto func3 = [this]() EXCLUSIVE_LOCK_FUNCTION(mu_) {
      mu_.Lock();
    };

    func1();  // expected-warning {{calling function 'operator()' requires holding mutex 'mu_' exclusively}}
    func2();
    func3();
    mu_.Unlock();
  }
};

}  // end namespace ThreadAttributesOnLambdas



namespace AttributeExpressionCornerCases {

class Foo {
  int a GUARDED_BY(getMu());

  Mutex* getMu()   LOCK_RETURNED("");
  Mutex* getUniv() LOCK_RETURNED("*");

  void test1() {
    a = 0;
  }

  void test2() EXCLUSIVE_LOCKS_REQUIRED(getUniv()) {
    a = 0;
  }

  void foo(Mutex* mu) EXCLUSIVE_LOCKS_REQUIRED(mu);

  void test3() {
    foo(nullptr);
  }
};


class MapTest {
  struct MuCell { Mutex* mu; };

  MyMap<MyString, Mutex*> map;
  MyMap<MyString, MuCell> mapCell;

  int a GUARDED_BY(map["foo"]);
  int b GUARDED_BY(mapCell["foo"].mu);

  void test() {
    map["foo"]->Lock();
    a = 0;
    map["foo"]->Unlock();
  }

  void test2() {
    mapCell["foo"].mu->Lock();
    b = 0;
    mapCell["foo"].mu->Unlock();
  }
};


class PreciseSmartPtr {
  SmartPtr<Mutex> mu;
  int val GUARDED_BY(mu);

  static bool compare(PreciseSmartPtr& a, PreciseSmartPtr &b) {
    a.mu->Lock();
    bool result = (a.val == b.val);   // expected-warning {{reading variable 'val' requires holding mutex 'b.mu'}} \
                                      // expected-note {{found near match 'a.mu'}}
    a.mu->Unlock();
    return result;
  }
};


class SmartRedeclare {
  SmartPtr<Mutex> mu;
  int val GUARDED_BY(mu);

  void test()  EXCLUSIVE_LOCKS_REQUIRED(mu);
  void test2() EXCLUSIVE_LOCKS_REQUIRED(mu.get());
  void test3() EXCLUSIVE_LOCKS_REQUIRED(mu.get());
};


void SmartRedeclare::test() EXCLUSIVE_LOCKS_REQUIRED(mu.get()) {
  val = 0;
}

void SmartRedeclare::test2() EXCLUSIVE_LOCKS_REQUIRED(mu) {
  val = 0;
}

void SmartRedeclare::test3() {
  val = 0;
}


namespace CustomMutex {


class LOCKABLE BaseMutex { };
class DerivedMutex : public BaseMutex { };

void customLock(const BaseMutex *m)   EXCLUSIVE_LOCK_FUNCTION(m);
void customUnlock(const BaseMutex *m) UNLOCK_FUNCTION(m);

static struct DerivedMutex custMu;

static void doSomethingRequiringLock() EXCLUSIVE_LOCKS_REQUIRED(custMu) { }

void customTest() {
  customLock(reinterpret_cast<BaseMutex*>(&custMu));  // ignore casts
  doSomethingRequiringLock();
  customUnlock(reinterpret_cast<BaseMutex*>(&custMu));
}

} // end namespace CustomMutex

} // end AttributeExpressionCornerCases


namespace ScopedLockReturnedInvalid {

class Opaque;

Mutex* getMutex(Opaque* o) LOCK_RETURNED("");

void test(Opaque* o) {
  MutexLock lock(getMutex(o));
}

}  // end namespace ScopedLockReturnedInvalid


namespace NegativeRequirements {

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
    mu.Lock();    // warning?  needs !mu?
    baz();        // expected-warning {{cannot call function 'baz' while mutex 'mu' is held}}
    bar();
    mu.Unlock();
  }

  void bar() {
    bar2();       // expected-warning {{calling function 'bar2' requires holding  '!mu'}}
  }

  void bar2() EXCLUSIVE_LOCKS_REQUIRED(!mu) {
    baz();
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
};

}   // end namespace NegativeRequirements


namespace NegativeThreadRoles {

typedef int __attribute__((capability("role"))) ThreadRole;

void acquire(ThreadRole R) __attribute__((exclusive_lock_function(R))) __attribute__((no_thread_safety_analysis)) {}
void release(ThreadRole R) __attribute__((unlock_function(R))) __attribute__((no_thread_safety_analysis)) {}

ThreadRole FlightControl, Logger;

extern void enque_log_msg(const char *msg);
void log_msg(const char *msg) {
  enque_log_msg(msg);
}

void dispatch_log(const char *msg) __attribute__((requires_capability(!FlightControl))) {}
void dispatch_log2(const char *msg) __attribute__((requires_capability(Logger))) {}

void flight_control_entry(void) __attribute__((requires_capability(FlightControl))) {
  dispatch_log("wrong"); /* expected-warning {{cannot call function 'dispatch_log' while mutex 'FlightControl' is held}} */
  dispatch_log2("also wrong"); /* expected-warning {{calling function 'dispatch_log2' requires holding role 'Logger' exclusively}} */
}

void spawn_fake_flight_control_thread(void) {
  acquire(FlightControl);
  flight_control_entry();
  release(FlightControl);
}

extern const char *deque_log_msg(void) __attribute__((requires_capability(Logger)));
void logger_entry(void) __attribute__((requires_capability(Logger))) {
  const char *msg;

  while ((msg = deque_log_msg())) {
    dispatch_log(msg);
  }
}

void spawn_fake_logger_thread(void) {
  acquire(Logger);
  logger_entry();
  release(Logger);
}

int main(void) {
  spawn_fake_flight_control_thread();
  spawn_fake_logger_thread();

  for (;;)
    ; /* Pretend to dispatch things. */

  return 0;
}

} // end namespace NegativeThreadRoles


namespace AssertSharedExclusive {

void doSomething();

class Foo {
  Mutex mu;
  int a GUARDED_BY(mu);

  void test() SHARED_LOCKS_REQUIRED(mu) {
    mu.AssertHeld();
    if (a > 0)
      doSomething();
  }
};

} // end namespace AssertSharedExclusive


namespace RangeBasedForAndReferences {

class Foo {
  struct MyStruct {
    int a;
  };

  Mutex mu;
  int a GUARDED_BY(mu);
  MyContainer<int>  cntr  GUARDED_BY(mu);
  MyStruct s GUARDED_BY(mu);
  int arr[10] GUARDED_BY(mu);

  void nonref_test() {
    int b = a;             // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
    b = 0;                 // no warning
  }

  void auto_test() {
    auto b = a;            // expected-warning {{reading variable 'a' requires holding mutex 'mu'}}
    b = 0;                 // no warning
    auto &c = a;           // no warning
    c = 0;                 // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
  }

  void ref_test() {
    int &b = a;
    int &c = b;
    int &d = c;
    b = 0;                 // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    c = 0;                 // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}
    d = 0;                 // expected-warning {{writing variable 'a' requires holding mutex 'mu' exclusively}}

    MyStruct &rs = s;
    rs.a = 0;              // expected-warning {{writing variable 's' requires holding mutex 'mu' exclusively}}

    int (&rarr)[10] = arr;
    rarr[2] = 0;           // expected-warning {{writing variable 'arr' requires holding mutex 'mu' exclusively}}
  }

  void ptr_test() {
    int *b = &a;
    *b = 0;                // no expected warning yet
  }

  void for_test() {
    int total = 0;
    for (int i : cntr) {   // expected-warning2 {{reading variable 'cntr' requires holding mutex 'mu'}}
      total += i;
    }
  }
};


} // end namespace RangeBasedForAndReferences



namespace PassByRefTest {

class Foo {
public:
  Foo() : a(0), b(0) { }

  int a;
  int b;

  void operator+(const Foo& f);

  void operator[](const Foo& g);
};

template<class T>
T&& mymove(T& f);


// test top-level functions
void copy(Foo f);
void write1(Foo& f);
void write2(int a, Foo& f);
void read1(const Foo& f);
void read2(int a, const Foo& f);
void destroy(Foo&& f);

void operator/(const Foo& f, const Foo& g);
void operator*(const Foo& f, const Foo& g);




class Bar {
public:
  Mutex mu;
  Foo           foo   GUARDED_BY(mu);
  Foo           foo2  GUARDED_BY(mu);
  Foo*          foop  PT_GUARDED_BY(mu);
  SmartPtr<Foo> foosp PT_GUARDED_BY(mu);

  // test methods.
  void mwrite1(Foo& f);
  void mwrite2(int a, Foo& f);
  void mread1(const Foo& f);
  void mread2(int a, const Foo& f);

  // static methods
  static void smwrite1(Foo& f);
  static void smwrite2(int a, Foo& f);
  static void smread1(const Foo& f);
  static void smread2(int a, const Foo& f);

  void operator<<(const Foo& f);

  void test1() {
    copy(foo);             // expected-warning {{reading variable 'foo' requires holding mutex 'mu'}}
    write1(foo);           // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    write2(10, foo);       // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    read1(foo);            // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    read2(10, foo);        // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    destroy(mymove(foo));  // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}

    mwrite1(foo);           // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    mwrite2(10, foo);       // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    mread1(foo);            // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    mread2(10, foo);        // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}

    smwrite1(foo);           // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    smwrite2(10, foo);       // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    smread1(foo);            // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}
    smread2(10, foo);        // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}

    foo + foo2;              // expected-warning {{reading variable 'foo' requires holding mutex 'mu'}} \
                             // expected-warning {{passing variable 'foo2' by reference requires holding mutex 'mu'}}
    foo / foo2;              // expected-warning {{reading variable 'foo' requires holding mutex 'mu'}} \
                             // expected-warning {{passing variable 'foo2' by reference requires holding mutex 'mu'}}
    foo * foo2;              // expected-warning {{reading variable 'foo' requires holding mutex 'mu'}} \
                             // expected-warning {{passing variable 'foo2' by reference requires holding mutex 'mu'}}
    foo[foo2];               // expected-warning {{reading variable 'foo' requires holding mutex 'mu'}} \
                             // expected-warning {{passing variable 'foo2' by reference requires holding mutex 'mu'}}
    (*this) << foo;          // expected-warning {{passing variable 'foo' by reference requires holding mutex 'mu'}}

    copy(*foop);             // expected-warning {{reading the value pointed to by 'foop' requires holding mutex 'mu'}}
    write1(*foop);           // expected-warning {{passing the value that 'foop' points to by reference requires holding mutex 'mu'}}
    write2(10, *foop);       // expected-warning {{passing the value that 'foop' points to by reference requires holding mutex 'mu'}}
    read1(*foop);            // expected-warning {{passing the value that 'foop' points to by reference requires holding mutex 'mu'}}
    read2(10, *foop);        // expected-warning {{passing the value that 'foop' points to by reference requires holding mutex 'mu'}}
    destroy(mymove(*foop));  // expected-warning {{passing the value that 'foop' points to by reference requires holding mutex 'mu'}}

    copy(*foosp);             // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}
    write1(*foosp);           // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}
    write2(10, *foosp);       // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}
    read1(*foosp);            // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}
    read2(10, *foosp);        // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}
    destroy(mymove(*foosp));  // expected-warning {{reading the value pointed to by 'foosp' requires holding mutex 'mu'}}

    // TODO -- these require better smart pointer handling.
    copy(*foosp.get());
    write1(*foosp.get());
    write2(10, *foosp.get());
    read1(*foosp.get());
    read2(10, *foosp.get());
    destroy(mymove(*foosp.get()));
  }
};


}  // end namespace PassByRefTest


namespace AcquiredBeforeAfterText {

class Foo {
  Mutex mu1 ACQUIRED_BEFORE(mu2, mu3);
  Mutex mu2;
  Mutex mu3;

  void test1() {
    mu1.Lock();
    mu2.Lock();
    mu3.Lock();

    mu3.Unlock();
    mu2.Unlock();
    mu1.Unlock();
  }

  void test2() {
    mu2.Lock();
    mu1.Lock();    // expected-warning {{mutex 'mu1' must be acquired before 'mu2'}}
    mu1.Unlock();
    mu2.Unlock();
  }

  void test3() {
    mu3.Lock();
    mu1.Lock();     // expected-warning {{mutex 'mu1' must be acquired before 'mu3'}}
    mu1.Unlock();
    mu3.Unlock();
  }

  void test4() EXCLUSIVE_LOCKS_REQUIRED(mu1) {
    mu2.Lock();
    mu2.Unlock();
  }

  void test5() EXCLUSIVE_LOCKS_REQUIRED(mu2) {
    mu1.Lock();    // expected-warning {{mutex 'mu1' must be acquired before 'mu2'}}
    mu1.Unlock();
  }

  void test6() EXCLUSIVE_LOCKS_REQUIRED(mu2) {
    mu1.AssertHeld();
  }

  void test7() EXCLUSIVE_LOCKS_REQUIRED(mu1, mu2, mu3) { }

  void test8() EXCLUSIVE_LOCKS_REQUIRED(mu3, mu2, mu1) { }
};


class Foo2 {
  Mutex mu1;
  Mutex mu2 ACQUIRED_AFTER(mu1);
  Mutex mu3 ACQUIRED_AFTER(mu1);

  void test1() {
    mu1.Lock();
    mu2.Lock();
    mu3.Lock();

    mu3.Unlock();
    mu2.Unlock();
    mu1.Unlock();
  }

  void test2() {
    mu2.Lock();
    mu1.Lock();     // expected-warning {{mutex 'mu1' must be acquired before 'mu2'}}
    mu1.Unlock();
    mu2.Unlock();
  }

  void test3() {
    mu3.Lock();
    mu1.Lock();     // expected-warning {{mutex 'mu1' must be acquired before 'mu3'}}
    mu1.Unlock();
    mu3.Unlock();
  }
};


class Foo3 {
  Mutex mu1 ACQUIRED_BEFORE(mu2);
  Mutex mu2;
  Mutex mu3 ACQUIRED_AFTER(mu2) ACQUIRED_BEFORE(mu4);
  Mutex mu4;

  void test1() {
    mu1.Lock();
    mu2.Lock();
    mu3.Lock();
    mu4.Lock();

    mu4.Unlock();
    mu3.Unlock();
    mu2.Unlock();
    mu1.Unlock();
  }

  void test2() {
    mu4.Lock();
    mu2.Lock();     // expected-warning {{mutex 'mu2' must be acquired before 'mu4'}}

    mu2.Unlock();
    mu4.Unlock();
  }

  void test3() {
    mu4.Lock();
    mu1.Lock();     // expected-warning {{mutex 'mu1' must be acquired before 'mu4'}}

    mu1.Unlock();
    mu4.Unlock();
  }

  void test4() {
    mu3.Lock();
    mu1.Lock();     // expected-warning {{mutex 'mu1' must be acquired before 'mu3'}}

    mu1.Unlock();
    mu3.Unlock();
  }
};


// Test transitive DAG traversal with AFTER
class Foo4 {
  Mutex mu1;
  Mutex mu2 ACQUIRED_AFTER(mu1);
  Mutex mu3 ACQUIRED_AFTER(mu1);
  Mutex mu4 ACQUIRED_AFTER(mu2, mu3);
  Mutex mu5 ACQUIRED_AFTER(mu4);
  Mutex mu6 ACQUIRED_AFTER(mu4);
  Mutex mu7 ACQUIRED_AFTER(mu5, mu6);
  Mutex mu8 ACQUIRED_AFTER(mu7);

  void test() {
    mu8.Lock();
    mu1.Lock();    // expected-warning {{mutex 'mu1' must be acquired before 'mu8'}}
    mu1.Unlock();
    mu8.Unlock();
  }
};


// Test transitive DAG traversal with BEFORE
class Foo5 {
  Mutex mu1 ACQUIRED_BEFORE(mu2, mu3);
  Mutex mu2 ACQUIRED_BEFORE(mu4);
  Mutex mu3 ACQUIRED_BEFORE(mu4);
  Mutex mu4 ACQUIRED_BEFORE(mu5, mu6);
  Mutex mu5 ACQUIRED_BEFORE(mu7);
  Mutex mu6 ACQUIRED_BEFORE(mu7);
  Mutex mu7 ACQUIRED_BEFORE(mu8);
  Mutex mu8;

  void test() {
    mu8.Lock();
    mu1.Lock();  // expected-warning {{mutex 'mu1' must be acquired before 'mu8'}}
    mu1.Unlock();
    mu8.Unlock();
  }
};


class Foo6 {
  Mutex mu1 ACQUIRED_AFTER(mu3);     // expected-warning {{Cycle in acquired_before/after dependencies, starting with 'mu1'}}
  Mutex mu2 ACQUIRED_AFTER(mu1);     // expected-warning {{Cycle in acquired_before/after dependencies, starting with 'mu2'}}
  Mutex mu3 ACQUIRED_AFTER(mu2);     // expected-warning {{Cycle in acquired_before/after dependencies, starting with 'mu3'}}

  Mutex mu_b ACQUIRED_BEFORE(mu_b);  // expected-warning {{Cycle in acquired_before/after dependencies, starting with 'mu_b'}}
  Mutex mu_a ACQUIRED_AFTER(mu_a);   // expected-warning {{Cycle in acquired_before/after dependencies, starting with 'mu_a'}}

  void test0() {
    mu_a.Lock();
    mu_b.Lock();
    mu_b.Unlock();
    mu_a.Unlock();
  }

  void test1a() {
    mu1.Lock();
    mu1.Unlock();
  }

  void test1b() {
    mu1.Lock();
    mu_a.Lock();
    mu_b.Lock();
    mu_b.Unlock();
    mu_a.Unlock();
    mu1.Unlock();
  }

  void test() {
    mu2.Lock();
    mu2.Unlock();
  }

  void test3() {
    mu3.Lock();
    mu3.Unlock();
  }
};

}  // end namespace AcquiredBeforeAfterTest


namespace ScopedAdoptTest {

class Foo {
  Mutex mu;
  int a GUARDED_BY(mu);
  int b;

  void test1() EXCLUSIVE_UNLOCK_FUNCTION(mu) {
    MutexLock slock(&mu, true);
    a = 0;
  }

  void test2() SHARED_UNLOCK_FUNCTION(mu) {
    ReaderMutexLock slock(&mu, true);
    b = a;
  }

  void test3() EXCLUSIVE_LOCKS_REQUIRED(mu) {  // expected-note {{mutex acquired here}}
    MutexLock slock(&mu, true);
    a = 0;
  }  // expected-warning {{expecting mutex 'mu' to be held at the end of function}}

  void test4() SHARED_LOCKS_REQUIRED(mu) {     // expected-note {{mutex acquired here}}
    ReaderMutexLock slock(&mu, true);
    b = a;
  }  // expected-warning {{expecting mutex 'mu' to be held at the end of function}}

};

}  // end namespace ScopedAdoptTest


namespace TestReferenceNoThreadSafetyAnalysis {

#define TS_UNCHECKED_READ(x) ts_unchecked_read(x)

// Takes a reference to a guarded data member, and returns an unguarded
// reference.
template <class T>
inline const T& ts_unchecked_read(const T& v) NO_THREAD_SAFETY_ANALYSIS {
  return v;
}

template <class T>
inline T& ts_unchecked_read(T& v) NO_THREAD_SAFETY_ANALYSIS {
  return v;
}


class Foo {
public:
  Foo(): a(0) { }

  int a;
};


class Bar {
public:
  Bar() : a(0) { }

  Mutex mu;
  int a   GUARDED_BY(mu);
  Foo foo GUARDED_BY(mu);
};


void test() {
  Bar bar;
  const Bar cbar;

  int a = TS_UNCHECKED_READ(bar.a);       // nowarn
  TS_UNCHECKED_READ(bar.a) = 1;           // nowarn

  int b = TS_UNCHECKED_READ(bar.foo).a;   // nowarn
  TS_UNCHECKED_READ(bar.foo).a = 1;       // nowarn

  int c = TS_UNCHECKED_READ(cbar.a);      // nowarn
}

#undef TS_UNCHECKED_READ

}  // end namespace TestReferenceNoThreadSafetyAnalysis


namespace GlobalAcquiredBeforeAfterTest {

Mutex mu1;
Mutex mu2 ACQUIRED_AFTER(mu1);

void test3() {
  mu2.Lock();
  mu1.Lock();  // expected-warning {{mutex 'mu1' must be acquired before 'mu2'}}
  mu1.Unlock();
  mu2.Unlock();
}

}  // end namespace  GlobalAcquiredBeforeAfterTest


namespace LifetimeExtensionText {

struct Holder {
  virtual ~Holder() throw() {}
  int i = 0;
};

void test() {
  // Should not crash.
  const auto &value = Holder().i;
}

} // end namespace LifetimeExtensionTest


namespace LockableUnions {

union LOCKABLE MutexUnion {
  int a;
  char* b;

  void Lock()   EXCLUSIVE_LOCK_FUNCTION();
  void Unlock() UNLOCK_FUNCTION();
};

MutexUnion muun2;
MutexUnion muun1 ACQUIRED_BEFORE(muun2);

void test() {
  muun2.Lock();
  muun1.Lock();  // expected-warning {{mutex 'muun1' must be acquired before 'muun2'}}
  muun1.Unlock();
  muun2.Unlock();
}

}  // end namespace LockableUnions

// This used to crash.
class acquired_before_empty_str {
  void WaitUntilSpaceAvailable() {
    lock_.ReaderLock(); // expected-note {{acquired here}}
  } // expected-warning {{mutex 'lock_' is still held at the end of function}}
  Mutex lock_ ACQUIRED_BEFORE("");
};

namespace PR34800 {
struct A {
  operator int() const;
};
struct B {
  bool g() __attribute__((locks_excluded(h))); // expected-warning {{'locks_excluded' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'int'}}
  int h;
};
struct C {
  B *operator[](int);
};
C c;
void f() { c[A()]->g(); }
} // namespace PR34800

namespace ReturnScopedLockable {
  template<typename Object> class SCOPED_LOCKABLE ReadLockedPtr {
  public:
    ReadLockedPtr(Object *ptr) SHARED_LOCK_FUNCTION((*this)->mutex);
    ReadLockedPtr(ReadLockedPtr &&) SHARED_LOCK_FUNCTION((*this)->mutex);
    ~ReadLockedPtr() UNLOCK_FUNCTION();

    Object *operator->() const { return object; }

  private:
    Object *object;
  };

  struct Object {
    int f() SHARED_LOCKS_REQUIRED(mutex);
    Mutex mutex;
  };

  ReadLockedPtr<Object> get();
  int use() {
    auto ptr = get();
    return ptr->f();
  }
}
