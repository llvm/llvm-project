// RUN: %clang_analyze_cc1 \
// RUN:   -analyzer-checker=alpha.unix.BlockInCriticalSection \
// RUN:   -std=c++11 \
// RUN:   -analyzer-output text \
// RUN:   -verify %s

void sleep(int x) {}

namespace std {
struct mutex {
  void lock() {}
  void unlock() {}
};
template<typename T>
struct lock_guard {
  lock_guard<T>(std::mutex) {}
  ~lock_guard<T>() {}
};
template<typename T>
struct unique_lock {
  unique_lock<T>(std::mutex) {}
  ~unique_lock<T>() {}
};
template<typename T>
struct not_real_lock {
  not_real_lock<T>(std::mutex) {}
};
} // namespace std

struct FILE;
int getc(FILE *stream);
char* fgets(char *str, FILE *stream);
using ssize_t = long long;
using size_t = unsigned long long;
ssize_t read(int fd, void *buf, size_t count);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);

struct pthread_mutex_t;
void pthread_mutex_lock(pthread_mutex_t *mutex);
void pthread_mutex_trylock(pthread_mutex_t *mutex);
void pthread_mutex_unlock(pthread_mutex_t *mutex);

struct mtx_t;
void mtx_lock(mtx_t *mutex);
void mtx_timedlock(mtx_t *mutex);
void mtx_trylock(mtx_t *mutex);
void mtx_unlock(mtx_t *mutex);

// global params for dummy function calls
FILE *stream;
char *str;
int fd;
void *buf;
size_t count;
int sockfd;
size_t len;
int flags;

void testBlockInCriticalSectionWithStdMutex() {
  std::mutex m;
  m.lock(); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  m.unlock();
}

void testBlockInCriticalSectionWithPthreadMutex(pthread_mutex_t *mutex) {
  pthread_mutex_lock(mutex); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock(mutex);

  pthread_mutex_trylock(mutex); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  pthread_mutex_unlock(mutex);
}

void testBlockInCriticalSectionC11Locks(mtx_t *mutex) {
  mtx_lock(mutex); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock(mutex);

  mtx_timedlock(mutex); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock(mutex);

  mtx_trylock(mutex); // expected-note 5{{Entering critical section here}}
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  getc(stream); // expected-warning {{Call to blocking function 'getc' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'getc' inside of critical section}}
  fgets(str, stream); // expected-warning {{Call to blocking function 'fgets' inside of critical section}}
           // expected-note@-1 {{Call to blocking function 'fgets' inside of critical section}}
  read(fd, buf, count); // expected-warning {{Call to blocking function 'read' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'read' inside of critical section}}
  recv(sockfd, buf, count, flags); // expected-warning {{Call to blocking function 'recv' inside of critical section}}
          // expected-note@-1 {{Call to blocking function 'recv' inside of critical section}}
  mtx_unlock(mutex);
}

void testMultipleBlockingCalls() {
  std::mutex m;
  m.lock(); // expected-note 1{{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  sleep(2); // no-warning
}

void testMultipleMutexesMultipleBlockingCalls() {
  std::mutex m, n, k;
  m.lock(); // expected-note 2{{Entering critical section here}}
  n.lock(); // expected-note 2{{Entering critical section here}}
  k.lock(); // expected-note 1{{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  k.unlock();
  sleep(2); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}


void testRecursiveAcquisition() {
  std::mutex m;
  m.lock(); // expected-note {{Entering critical section for the 1st time here}}
  m.lock(); // expected-note {{Entering critical section for the 2nd time here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  m.unlock();
}

void testRecursiveAcquisitionWithMultipleBlockingCalls() {
  std::mutex m;
  m.lock(); // expected-note 1{{Entering critical section for the 1st time here}}
            // expected-note@-1 {{Entering critical section here}}
  m.lock(); // expected-note 1{{Entering critical section for the 2nd time here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  // this next 'sleep' call is only in the critical section of the first lock
  sleep(2); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
}

void testRecursiveAcquisitionWithMultipleMutexes() {
  std::mutex m, n;
  m.lock(); // expected-note 1{{Entering critical section here}}
  n.lock(); // expected-note 2{{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  // this next 'sleep' call is only in the critical section of mutex 'n'
  sleep(2); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  n.unlock();
}


void testNestedMutexes() {
  std::mutex m, n, k;
  m.lock(); // expected-note 3{{Entering critical section here}}
  n.lock(); // expected-note 2{{Entering critical section here}}
  k.lock(); // expected-note 1{{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  k.unlock();
  sleep(2); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  n.unlock();
  sleep(3); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
  sleep(4); // no-warning
}

void testNonOverlappingMutexes() {
  std::mutex m;
  m.lock(); // There should be no warning here
  m.unlock();
  m.lock(); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
}

void testMixedMutexLocksWithIntermittentUnlock() {
  std::mutex m, n, k;
  m.lock(); // expected-note {{Entering critical section here}}
  n.lock(); // the problem is not is this lock's critical section
  n.unlock();
  k.lock(); // same as for n.lock()
  k.unlock();
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
  m.unlock();
}

void f() {
  sleep(1000); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
               // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionInterProcedural() {
  std::mutex m;
  m.lock(); // expected-note {{Entering critical section here}}
  f(); // expected-note {{Calling 'f'}}
  m.unlock();
}

void unknown_function_that_may_lock(std::mutex &);
void testBlockInCriticalSectionUnexpectedUnlock() {
  std::mutex m;
  unknown_function_that_may_lock(m);
  m.unlock();
  sleep(1); // no-warning
  m.lock(); // expected-note {{Entering critical section here}}
  sleep(2); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionLockGuard() {
  std::mutex g_mutex;
  std::not_real_lock<std::mutex> not_real_lock(g_mutex);
  sleep(1); // no-warning

  std::lock_guard<std::mutex> lock(g_mutex); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionLockGuardNested() {
  testBlockInCriticalSectionLockGuard(); // expected-note {{Calling 'testBlockInCriticalSectionLockGuard'}}
  sleep(1); // no-warning
}

void testBlockInCriticalSectionUniqueLock() {
  std::mutex g_mutex;
  std::not_real_lock<std::mutex> not_real_lock(g_mutex);
  sleep(1); // no-warning

  std::unique_lock<std::mutex> lock(g_mutex); // expected-note {{Entering critical section here}}
  sleep(1); // expected-warning {{Call to blocking function 'sleep' inside of critical section}}
            // expected-note@-1 {{Call to blocking function 'sleep' inside of critical section}}
}

void testBlockInCriticalSectionUniqueLockNested() {
  testBlockInCriticalSectionUniqueLock(); // expected-note {{Calling 'testBlockInCriticalSectionUniqueLock'}}
  sleep(1); // no-warning
}
