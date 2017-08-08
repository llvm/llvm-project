// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.unix.BlockInCriticalSection -std=c++11 -verify %s

void sleep(int x) {}

namespace std {
struct mutex {
  void lock() {}
  void unlock() {}
};
}

void getc() {}
void fgets() {}
void read() {}
void recv() {}

void pthread_mutex_lock() {}
void pthread_mutex_trylock() {}
void pthread_mutex_unlock() {}

void mtx_lock() {}
void mtx_timedlock() {}
void mtx_trylock() {}
void mtx_unlock() {}

void testBlockInCriticalSectionWithStdMutex() {
  std::mutex m;
  m.lock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  m.unlock();
}

void testBlockInCriticalSectionWithPthreadMutex() {
  pthread_mutex_lock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock();

  pthread_mutex_trylock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock();
}

void testBlockInCriticalSectionC11Locks() {
  mtx_lock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();

  mtx_timedlock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();

  mtx_trylock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  getc(); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
  fgets(); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
  read(); // expected-warning {{Call to blocking function 'read' inside of critical section}}
  recv(); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock();
}

void testBlockInCriticalSectionWithNestedMutexes() {
  std::mutex m, n, k;
  m.lock();
  n.lock();
  k.lock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  k.unlock();
  sleep(5); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  n.unlock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  sleep(3); // no-warning
}

void f() {
  sleep(1000); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionInterProcedural() {
  std::mutex m;
  m.lock();
  f();
  m.unlock();
}

void testBlockInCriticalSectionUnexpectedUnlock() {
  std::mutex m;
  m.unlock();
  sleep(1); // no-warning
  m.lock();
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
}
