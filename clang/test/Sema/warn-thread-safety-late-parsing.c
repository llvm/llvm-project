// Under -fexperimental-late-parse-attributes, capability attributes are late
// parsed, so they can name a member declared later in the same struct or a
// parameter declared later in the same prototype. Without it, those forward
// references are errors.
//
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify=both %s
// RUN: %clang_cc1 -fsyntax-only -verify=both,early %s

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

// guarded_by behaves the same as the rest of the family.
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

// An attribute on a parameter may name another parameter of the same prototype,
// including one declared later.
void put_later(void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
               struct Mutex *mu);

// Naming an earlier parameter needs no late parsing; it works in both modes.
void put_earlier(struct Mutex *mu, void (*release)(int) RELEASE(mu));

// The requirement may also name a member reached through a later parameter.
struct Holder {
  struct Mutex mu;
};
void put_member(void (*release)(int) RELEASE(&h->mu), // early-error{{use of undeclared identifier 'h'}}
                struct Holder *h);

// A parameter of the pointee type is still resolved, in both modes.
void pointee_param(void (*release)(struct Holder *inner) RELEASE(&inner->mu));

// A parameter declared with a function type is adjusted to a function pointer;
// both a later parameter and a pointee parameter resolve the same way.
void put_later_decayed(void release(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                       struct Mutex *mu);
void pointee_decayed(void release(struct Holder *inner) RELEASE(&inner->mu));

// One attribute may name both a pointee parameter and a later sibling member.
struct Both {
  void (*cb)(struct Mutex *pm) REQUIRES(pm, sm); // early-error{{use of undeclared identifier 'sm'}}
  struct Mutex sm;
};

// For nested function declarators, the parameters of the innermost prototype
// are the ones in scope, in both modes.
void nested(void (*(*f)(struct Mutex *m))(int) RELEASE(m));

// A pointee parameter shadows a later member or parameter of the same name, and
// may be named together with a later parameter. Which declaration each name
// binds to is checked in warn-thread-safety-analysis.c.
struct ShadowMember {
  void (*cb)(struct Holder *mu) REQUIRES(&mu->mu);
  struct Mutex mu;
};
void shadow_param(void (*release)(struct Holder *mu) RELEASE(&mu->mu),
                  struct Mutex *mu);
void pointee_and_later(void (*release)(struct Holder *h) RELEASE(&h->mu, mu), // early-error{{use of undeclared identifier 'mu'}}
                       struct Mutex *mu);

// A pointee's parameters are in scope only for its own attributes.
struct Leak {
  void (*cb)(struct Mutex *p) REQUIRES(p);
  int x GUARDED_BY(p); // both-error{{use of undeclared identifier 'p'}}
};
void leak(void (*r)(struct Mutex *p) RELEASE(p),
          void (*s)(int) RELEASE(p)); // both-error{{use of undeclared identifier 'p'}}
struct Multi {
  void (*a)(struct Mutex *p) REQUIRES(p),
       (*b)(struct Mutex *q) REQUIRES(q, p); // both-error{{use of undeclared identifier 'p'}}
};

// Outside a struct, and with unnamed parameters alongside.
void (*global_cb)(struct Mutex *p) REQUIRES(p);
struct Unnamed {
  void (*cb)(int, struct Mutex *p, int) REQUIRES(p);
};
