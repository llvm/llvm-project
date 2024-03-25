// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta %s

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__ ((assert_exclusive_lock(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__ ((assert_shared_lock(__VA_ARGS__)))
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

// Define the mutex struct.
// Simplified only for test purpose.
struct LOCKABLE Mutex {};

struct Foo {
  struct Mutex *mu_;
};

// Declare mutex lock/unlock functions.
void mutex_exclusive_lock(struct Mutex *mu) EXCLUSIVE_LOCK_FUNCTION(mu);
void mutex_shared_lock(struct Mutex *mu) SHARED_LOCK_FUNCTION(mu);
void mutex_unlock(struct Mutex *mu) UNLOCK_FUNCTION(mu);
void mutex_shared_unlock(struct Mutex *mu) __attribute__((release_shared_capability(mu)));
void mutex_exclusive_unlock(struct Mutex *mu) __attribute__((release_capability(mu)));

// Define global variables.
struct Mutex mu1;
struct Mutex mu2 ACQUIRED_AFTER(mu1);
struct Foo foo_ = {&mu1};
int a_ GUARDED_BY(foo_.mu_);
int *b_ PT_GUARDED_BY(foo_.mu_) = &a_;
int c_ GUARDED_VAR;
int *d_ PT_GUARDED_VAR = &c_;

// Define test functions.
int Foo_fun1(int i) SHARED_LOCKS_REQUIRED(mu2) EXCLUSIVE_LOCKS_REQUIRED(mu1) {
  return i;
}

int Foo_fun2(int i) EXCLUSIVE_LOCKS_REQUIRED(mu2) SHARED_LOCKS_REQUIRED(mu1) {
  return i;
}

int Foo_func3(int i) LOCKS_EXCLUDED(mu1, mu2) {
  return i;
}

static int Bar_fun1(int i) EXCLUSIVE_LOCKS_REQUIRED(mu1) {
  return i;
}

void set_value(int *a, int value) EXCLUSIVE_LOCKS_REQUIRED(foo_.mu_) {
  *a = value;
}

int get_value(int *p) SHARED_LOCKS_REQUIRED(foo_.mu_){
  return *p;
}

void unlock_scope(struct Mutex *const *mu) __attribute__((release_capability(**mu)));

int main(void) {

  Foo_fun1(1); // expected-warning{{calling function 'Foo_fun1' requires holding mutex 'mu2'}} \
                  expected-warning{{calling function 'Foo_fun1' requires holding mutex 'mu1' exclusively}}

  mutex_exclusive_lock(&mu1); // expected-note{{mutex acquired here}}
  mutex_shared_lock(&mu2);
  Foo_fun1(1);

  mutex_shared_lock(&mu1); // expected-warning{{acquiring mutex 'mu1' that is already held}} \
                              expected-warning{{mutex 'mu1' must be acquired before 'mu2'}}
  mutex_unlock(&mu1);
  mutex_unlock(&mu2);
  mutex_shared_lock(&mu1);
  mutex_exclusive_lock(&mu2);
  Foo_fun2(2);

  mutex_unlock(&mu2);
  mutex_unlock(&mu1);
  mutex_exclusive_lock(&mu1);
  Bar_fun1(3);
  mutex_unlock(&mu1);

  mutex_exclusive_lock(&mu1);
  Foo_func3(4);  // expected-warning{{cannot call function 'Foo_func3' while mutex 'mu1' is held}}
  mutex_unlock(&mu1);

  Foo_func3(5);

  set_value(&a_, 0); // expected-warning{{calling function 'set_value' requires holding mutex 'foo_.mu_' exclusively}}
  get_value(b_); // expected-warning{{calling function 'get_value' requires holding mutex 'foo_.mu_'}}
  mutex_exclusive_lock(foo_.mu_);
  set_value(&a_, 1);
  mutex_unlock(foo_.mu_);
  mutex_shared_lock(foo_.mu_);
  (void)(get_value(b_) == 1);
  mutex_unlock(foo_.mu_);

  c_ = 0; // expected-warning{{writing variable 'c_' requires holding any mutex exclusively}}
  (void)(*d_ == 0); // expected-warning{{reading the value pointed to by 'd_' requires holding any mutex}}
  mutex_exclusive_lock(foo_.mu_);
  c_ = 1;
  (void)(*d_ == 1);
  mutex_unlock(foo_.mu_);

  mutex_exclusive_lock(&mu1);    // expected-note {{mutex acquired here}}
  mutex_shared_unlock(&mu1);     // expected-warning {{releasing mutex 'mu1' using shared access, expected exclusive access}}
                                 // expected-note@-1{{mutex released here}}
  mutex_exclusive_unlock(&mu1);  // expected-warning {{releasing mutex 'mu1' that was not held}}

  mutex_shared_lock(&mu1);      // expected-note {{mutex acquired here}}
  mutex_exclusive_unlock(&mu1); // expected-warning {{releasing mutex 'mu1' using exclusive access, expected shared access}}
                                // expected-note@-1{{mutex released here}}
  mutex_shared_unlock(&mu1);    // expected-warning {{releasing mutex 'mu1' that was not held}}

  /// Cleanup functions
  {
    struct Mutex* const __attribute__((cleanup(unlock_scope))) scope = &mu1;
    mutex_exclusive_lock(scope);  // Note that we have to lock through scope, because no alias analysis!
    // Cleanup happens automatically -> no warning.
  }

  return 0;
}

// We had a problem where we'd skip all attributes that follow a late-parsed
// attribute in a single __attribute__.
void run(void) __attribute__((guarded_by(mu1), guarded_by(mu1))); // expected-warning 2{{only applies to non-static data members and global variables}}
