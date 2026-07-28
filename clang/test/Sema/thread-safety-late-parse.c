// Capability attributes are late parsed under -fexperimental-late-parse-attributes,
// like guarded_by and pt_guarded_by already were, so they can name a member
// declared later in the same struct.
//
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify=late %s
// RUN: %clang_cc1 -fsyntax-only -verify=early %s

// late-no-diagnostics

#define REQUIRES(...)     __attribute__((requires_capability(__VA_ARGS__)))
#define ACQUIRE(...)      __attribute__((acquire_capability(__VA_ARGS__)))
#define RELEASE(...)      __attribute__((release_capability(__VA_ARGS__)))
#define ASSERT_CAP(...)   __attribute__((assert_capability(__VA_ARGS__)))
#define TRY_ACQUIRE(...)  __attribute__((try_acquire_capability(__VA_ARGS__)))
#define EXCLUDES(...)     __attribute__((locks_excluded(__VA_ARGS__)))
#define RETURN_CAP(x)     __attribute__((lock_returned(x)))
#define GUARDED_BY(x)     __attribute__((guarded_by(x)))

struct __attribute__((capability("mutex"))) Mutex {
  int dummy;
};

struct Requires {
  void (*cb)(void) REQUIRES(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

struct Acquire {
  void (*cb)(void) ACQUIRE(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

struct Release {
  void (*cb)(void) RELEASE(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

struct Assert {
  void (*cb)(void) ASSERT_CAP(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

struct TryAcquire {
  int (*cb)(void) TRY_ACQUIRE(1, mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

struct Excludes {
  void (*cb)(void) EXCLUDES(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

// guarded_by was already late parsed; it is here to show the family now agrees.
struct Guarded {
  int data GUARDED_BY(mu); // early-error{{use of undeclared identifier 'mu'}}
  struct Mutex mu;
};

// An attribute after a complete parameter list already sees those parameters
// without late parsing; this must keep working in both modes.
struct WithGetter {
  struct Mutex mu;
};
struct Mutex *get_mu(struct WithGetter *w) RETURN_CAP(w->mu);
void use_getter(struct WithGetter *w) REQUIRES(get_mu(w));
