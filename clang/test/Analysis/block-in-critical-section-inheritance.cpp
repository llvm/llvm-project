// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

unsigned int sleep(unsigned int seconds) {return 0;}
namespace std {
// There are some standard library implementations where some mutex methods
// come from an implementation detail base class. We need to ensure that these
// are matched correctly.
class __mutex_base {
public:
  void lock();
};
class mutex : public __mutex_base{
public:
  void unlock();
  bool try_lock();
};
} // namespace std

void gh_99628() {
  std::mutex m;
  m.lock();
  // expected-note@-1 {{Entering critical section here}}
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
}

void no_false_positive_gh_104241() {
  std::mutex m;
  m.lock();
  // If inheritance not handled properly, this unlock might not match the lock
  // above because technically they act on different memory regions:
  // __mutex_base and mutex.
  m.unlock();
  sleep(10); // no-warning
}

struct TwoMutexes {
  std::mutex m1;
  std::mutex m2;
};

void two_mutexes_no_false_negative(TwoMutexes &tm) {
  tm.m1.lock();
  // expected-note@-1 {{Entering critical section here}}
  tm.m2.unlock();
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  tm.m1.unlock();
}

struct MyMutexBase1 : std::mutex {
  void lock1() { lock(); }
  // expected-note@-1 {{Entering critical section here}}
  void unlock1() { unlock(); }
};
struct MyMutexBase2 : std::mutex {
  void lock2() { lock(); }
  void unlock2() { unlock(); }
};
struct MyMutex : MyMutexBase1, MyMutexBase2 {};
// MyMutex has two distinct std::mutex as base classes

void custom_mutex_tp(MyMutexBase1 &mb) {
  mb.lock();
  // expected-note@-1 {{Entering critical section here}}
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  mb.unlock();
}

void custom_mutex_tn(MyMutexBase1 &mb) {
  mb.lock();
  mb.unlock();
  sleep(10);
}

void custom_mutex_cast_tp(MyMutexBase1 &mb) {
  static_cast<std::mutex&>(mb).lock();
  // expected-note@-1 {{Entering critical section here}}
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  static_cast<std::mutex&>(mb).unlock();
}

void custom_mutex_cast_tn(MyMutexBase1 &mb) {
  static_cast<std::mutex&>(mb).lock();
  static_cast<std::mutex&>(mb).unlock();
  sleep(10);
}

void two_custom_mutex_bases_tp(MyMutex &m) {
  m.lock1();
  // expected-note@-1 {{Calling 'MyMutexBase1::lock1'}}
  // expected-note@-2 {{Returning from 'MyMutexBase1::lock1'}}
  m.unlock2();
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock1();
}

void two_custom_mutex_bases_tn(MyMutex &m) {
  m.lock1();
  m.unlock1();
  sleep(10);
}

void two_custom_mutex_bases_casts_tp(MyMutex &m) {
  static_cast<MyMutexBase1&>(m).lock();
  // expected-note@-1 {{Entering critical section here}}
  static_cast<MyMutexBase2&>(m).unlock();
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  static_cast<MyMutexBase1&>(m).unlock();
}

void two_custom_mutex_bases_casts_tn(MyMutex &m) {
  static_cast<MyMutexBase1&>(m).lock();
  static_cast<MyMutexBase1&>(m).unlock();
  sleep(10);
}

struct MutexVirtBase1 : virtual std::mutex {
  void lock1() { lock(); }
  // expected-note@-1 {{Entering critical section here}}
  void unlock1() { unlock(); }
};

struct MutexVirtBase2 : virtual std::mutex {
  void lock2() { lock(); }
  void unlock2() { unlock(); }
};

struct CombinedVirtMutex : MutexVirtBase1, MutexVirtBase2 {};

void virt_inherited_mutexes_same_base_tn1(CombinedVirtMutex &cvt) {
  cvt.lock1();
  cvt.unlock1();
  sleep(10);
}

void virt_inherited_mutexes_different_bases_tn(CombinedVirtMutex &cvt) {
  cvt.lock1();
  cvt.unlock2(); // Despite a different name, unlock2 acts on the same mutex as lock1
  sleep(10);
}

void virt_inherited_mutexes_different_bases_tp(CombinedVirtMutex &cvt) {
  cvt.lock1();
  // expected-note@-1 {{Calling 'MutexVirtBase1::lock1'}}
  // expected-note@-2 {{Returning from 'MutexVirtBase1::lock1'}}
  sleep(10);
  // expected-warning@-1 {{Call to blocking function 'sleep' inside of critical section}}
  // expected-note@-2 {{Call to blocking function 'sleep' inside of critical section}}
  cvt.unlock1();
}

namespace std {
template <class... MutexTypes> struct scoped_lock {
  explicit scoped_lock(MutexTypes&... m);
  ~scoped_lock();
};
template <class MutexType> class scoped_lock<MutexType> {
public:
  explicit scoped_lock(MutexType& m) : m(m) { m.lock(); }
  ~scoped_lock() { m.unlock(); }
private:
  MutexType& m;
};
} // namespace std

namespace gh_104241 {
int magic_number;
std::mutex m;

void fixed() {
  int current;
  for (int items_processed = 0; items_processed < 100; ++items_processed) {
    {
      std::scoped_lock<std::mutex> guard(m);
      current = magic_number;
    }
    sleep(current); // expected no warning
  }
}
} // namespace gh_104241
