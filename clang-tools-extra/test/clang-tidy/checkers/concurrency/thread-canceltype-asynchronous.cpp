// RUN: %check_clang_tidy %s concurrency-thread-canceltype-asynchronous %t

#define ONE (1 << 0)

#define PTHREAD_CANCEL_DEFERRED 0
// define the macro intentionally complex
#define PTHREAD_CANCEL_ASYNCHRONOUS ONE

#define ASYNCHR PTHREAD_CANCEL_ASYNCHRONOUS

int pthread_setcanceltype(int type, int *oldtype);

int main() {
  int result, oldtype;

  if ((result = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype)) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: the cancel type for a pthread should not be 'PTHREAD_CANCEL_ASYNCHRONOUS' [concurrency-thread-canceltype-asynchronous]
    return 1;
  }

  if ((result = pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &oldtype)) != 0) {
    return 1;
  }

  return 0;
}

int f1() {
  int result, oldtype;

  if ((result = pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype)) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: the cancel type for a pthread should not be 'PTHREAD_CANCEL_ASYNCHRONOUS' [concurrency-thread-canceltype-asynchronous]
    return 1;
  }

  if ((result = pthread_setcanceltype(ASYNCHR, &oldtype)) != 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: the cancel type for a pthread should not be 'PTHREAD_CANCEL_ASYNCHRONOUS' [concurrency-thread-canceltype-asynchronous]
    return 1;
  }

  return 0;
}

int f2(int type) {
  int result, oldtype;

  if ((result = pthread_setcanceltype(type, &oldtype)) != 0) {
    return 1;
  }

  return 0;
}
