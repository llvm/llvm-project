// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta -fexperimental-late-parse-attributes -DLATE_PARSING %s

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(...)     __attribute__ ((guarded_by(__VA_ARGS__)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(...)  __attribute__ ((pt_guarded_by(__VA_ARGS__)))
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
  int  a_value GUARDED_BY(mu_);

  struct Bar {
    struct Mutex *other_mu ACQUIRED_AFTER(mu_); // Note: referencing the parent structure is convenient here, but this should probably be disallowed if the child structure is re-used outside of the parent.
    struct Mutex *third_mu ACQUIRED_BEFORE(other_mu);
  } bar;

  int* a_ptr PT_GUARDED_BY(bar.other_mu);
};

struct LOCKABLE Lock {};
struct A {
        struct Lock lock;
        union {
                int b __attribute__((guarded_by(lock))); // Note: referencing the parent structure is convenient here, but this should probably be disallowed if the child is re-used outside of the parent.
        };
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

// Verify late parsing:
#ifdef LATE_PARSING
struct LateParsing {
  int a_value_defined_before GUARDED_BY(a_mutex_defined_late);
  int *a_ptr_defined_before PT_GUARDED_BY(a_mutex_defined_late);
  struct Mutex *a_mutex_defined_early
    ACQUIRED_BEFORE(a_mutex_defined_late);
  struct Mutex *a_mutex_defined_late
    ACQUIRED_AFTER(a_mutex_defined_very_late);
  struct Mutex *a_mutex_defined_very_late;
} late_parsing;
#endif

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

  foo_.a_value = 0; // expected-warning {{writing variable 'a_value' requires holding mutex 'mu_' exclusively}}
  *foo_.a_ptr = 1; // expected-warning {{writing the value pointed to by 'a_ptr' requires holding mutex 'bar.other_mu' exclusively}}


  mutex_exclusive_lock(foo_.bar.other_mu);
  mutex_exclusive_lock(foo_.bar.third_mu); // expected-warning{{mutex 'third_mu' must be acquired before 'other_mu'}}
  mutex_exclusive_lock(foo_.mu_); // expected-warning{{mutex 'mu_' must be acquired before 'other_mu'}}
  mutex_exclusive_unlock(foo_.mu_);
  mutex_exclusive_unlock(foo_.bar.other_mu);
  mutex_exclusive_unlock(foo_.bar.third_mu);

#ifdef LATE_PARSING
  late_parsing.a_value_defined_before = 1; // expected-warning{{writing variable 'a_value_defined_before' requires holding mutex 'a_mutex_defined_late' exclusively}}
  late_parsing.a_ptr_defined_before = 0;
  mutex_exclusive_lock(late_parsing.a_mutex_defined_late);
  mutex_exclusive_lock(late_parsing.a_mutex_defined_early); // expected-warning{{mutex 'a_mutex_defined_early' must be acquired before 'a_mutex_defined_late'}}
  mutex_exclusive_unlock(late_parsing.a_mutex_defined_early);
  mutex_exclusive_unlock(late_parsing.a_mutex_defined_late);
  mutex_exclusive_lock(late_parsing.a_mutex_defined_late);
  mutex_exclusive_lock(late_parsing.a_mutex_defined_very_late); // expected-warning{{mutex 'a_mutex_defined_very_late' must be acquired before 'a_mutex_defined_late'}}
  mutex_exclusive_unlock(late_parsing.a_mutex_defined_very_late);
  mutex_exclusive_unlock(late_parsing.a_mutex_defined_late);
#endif

  return 0;
}

// We had a problem where we'd skip all attributes that follow a late-parsed
// attribute in a single __attribute__.
void run(void) __attribute__((guarded_by(mu1), guarded_by(mu1))); // expected-warning 2{{only applies to non-static data members and global variables}}

int value_with_wrong_number_of_args GUARDED_BY(mu1, mu2); // expected-error{{'guarded_by' attribute takes one argument}}

int *ptr_with_wrong_number_of_args PT_GUARDED_BY(mu1, mu2); // expected-error{{'pt_guarded_by' attribute takes one argument}}

int value_with_no_open_brace __attribute__((guarded_by)); // expected-error{{'guarded_by' attribute takes one argument}}
int *ptr_with_no_open_brace __attribute__((pt_guarded_by)); // expected-error{{'pt_guarded_by' attribute takes one argument}}

int value_with_no_open_brace_on_acquire_after __attribute__((acquired_after)); // expected-error{{'acquired_after' attribute takes at least 1 argument}}
int value_with_no_open_brace_on_acquire_before __attribute__((acquired_before)); // expected-error{{'acquired_before' attribute takes at least 1 argument}}

int value_with_bad_expr GUARDED_BY(bad_expr); // expected-error{{use of undeclared identifier 'bad_expr'}}
int *ptr_with_bad_expr PT_GUARDED_BY(bad_expr); // expected-error{{use of undeclared identifier 'bad_expr'}}

int value_with_bad_expr_on_acquire_after __attribute__((acquired_after(other_bad_expr))); //  expected-error{{use of undeclared identifier 'other_bad_expr'}}
int value_with_bad_expr_on_acquire_before __attribute__((acquired_before(other_bad_expr))); //  expected-error{{use of undeclared identifier 'other_bad_expr'}}

int a_final_expression = 0;
