// Test this without pch.
// RUN: %clang_cc1 -include %s -fsyntax-only -verify -Wthread-safety -std=c++11 %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s -std=c++11
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify -Wthread-safety -std=c++11 %s

#ifndef HEADER
#define HEADER

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)   __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)      __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__ ((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__ ((shared_trylock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock_function(__VA_ARGS__)))
#define LOCK_RETURNED(x)    __attribute__ ((lock_returned(x)))
#define LOCKS_EXCLUDED(...) __attribute__ ((locks_excluded(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  __attribute__ ((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...) \
  __attribute__ ((shared_locks_required(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS  __attribute__ ((no_thread_safety_analysis))


class  __attribute__((lockable)) Mutex {
 public:
  void Lock() __attribute__((exclusive_lock_function));
  void ReaderLock() __attribute__((shared_lock_function));
  void Unlock() __attribute__((unlock_function));
  bool TryLock() __attribute__((exclusive_trylock_function(true)));
  bool ReaderTryLock() __attribute__((shared_trylock_function(true)));
  void LockWhen(const int &cond) __attribute__((exclusive_lock_function));
};

class __attribute__((scoped_lockable)) MutexLock {
 public:
  MutexLock(Mutex *mu) __attribute__((exclusive_lock_function(mu)));
  ~MutexLock() __attribute__((unlock_function));
};

class __attribute__((scoped_lockable)) ReaderMutexLock {
 public:
  ReaderMutexLock(Mutex *mu) __attribute__((exclusive_lock_function(mu)));
  ~ReaderMutexLock() __attribute__((unlock_function));
};

class SCOPED_LOCKABLE ReleasableMutexLock {
 public:
  ReleasableMutexLock(Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu);
  ~ReleasableMutexLock() UNLOCK_FUNCTION();

  void Release() UNLOCK_FUNCTION();
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

private:
  T* ptr_;
};


// For testing destructor calls and cleanup.
class MyString {
public:
  MyString(const char* s);
  ~MyString();
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

#else

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
  sls_mu.Lock(); // expected-note{{mutex acquired here}}
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
  while (getBool()) { // \
        expected-warning{{expecting mutex 'sls_mu' to be held at start of each loop}}
    sls_mu.Unlock();
    if (getBool()) {
      if (getBool()) {
        continue;
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
        break;
      }
    }
    sls_mu.Lock();
  }
  sls_mu.Unlock(); // \
    expected-warning{{mutex 'sls_mu' is not held on every path through here}} \
    expected-warning{{releasing mutex 'sls_mu' that was not held}}
}

#endif
