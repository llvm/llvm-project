// RUN: %check_clang_tidy %s misc-non-copyable-objects %t

typedef struct FILE {} FILE;
typedef struct pthread_cond_t {} pthread_cond_t;
typedef int pthread_mutex_t;

// CHECK-MESSAGES: :[[@LINE+1]]:13: warning: 'f' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'? [misc-non-copyable-objects]
void g(FILE f);
// CHECK-MESSAGES: :[[@LINE+1]]:24: warning: 'm' declared as type 'pthread_mutex_t', which is unsafe to copy; did you mean 'pthread_mutex_t *'?
void h(pthread_mutex_t m);
// CHECK-MESSAGES: :[[@LINE+1]]:23: warning: 'c' declared as type 'pthread_cond_t', which is unsafe to copy; did you mean 'pthread_cond_t *'?
void i(pthread_cond_t c);

struct S {
  pthread_cond_t c; // ok
  // CHECK-MESSAGES: :[[@LINE+1]]:8: warning: 'f' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  FILE f;
};

void func(FILE *f) {
  // CHECK-MESSAGES: :[[@LINE+1]]:8: warning: 'f1' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  FILE f1; // match
  // CHECK-MESSAGES: :[[@LINE+2]]:8: warning: 'f2' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: expression has opaque data structure type 'FILE'; type should only be used as a pointer and not dereferenced
  FILE f2 = *f;
  // CHECK-MESSAGES: :[[@LINE+1]]:15: warning: 'f3' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  struct FILE f3;
  // CHECK-MESSAGES: :[[@LINE+1]]:16: warning: expression has opaque data structure type 'FILE'; type should only be used as a pointer and not dereferenced
  (void)sizeof(*f);
  (void)sizeof(FILE);
  // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: expression has opaque data structure type 'FILE'; type should only be used as a pointer and not dereferenced
  g(*f);

  pthread_mutex_t m; // ok
  h(m); // ok

  pthread_cond_t c; // ok
  i(c); // ok

  pthread_mutex_t *m1 = &m; // ok
  // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: expression has opaque data structure type 'pthread_mutex_t'; type should only be used as a pointer and not dereferenced
  h(*m1);
}
