// Under -fexperimental-late-parse-attributes, an attribute on a parameter may
// name a parameter declared later in the same prototype. Without it, that
// forward reference is an error.
//
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify=both -Wthread-safety -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify=both,early -Wthread-safety -std=c++20 %s

#define RELEASE(...)  __attribute__((release_capability(__VA_ARGS__)))
#define REQUIRES(...) __attribute__((requires_capability(__VA_ARGS__)))

class __attribute__((capability("mutex"))) Mutex {};

struct Holder {
  Mutex lock;
};

void put_later(void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
               Mutex *mu);

// A lambda's parameter clause is parsed the same way.
auto lambda = [](void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                 Mutex *mu) {};

struct Methods {
  void put_later(void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                 Mutex *mu);
  // A class's attributes are late parsed in C++ regardless of the flag; one may
  // name a pointee parameter and a member of the class together.
  void (*cb)(Holder *h) REQUIRES(h->lock, own);
  Mutex own;
};


void default_arg(void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                 Mutex *mu = nullptr);

// A class's attributes were always late parsed, when a pointee's parameters
// were not in scope, so a name that would now bind to one instead of to a
// member or a global is ambiguous rather than silently rebound.
Mutex global_mu; // both-note{{candidate found by name lookup is 'global_mu'}}
struct Scoping {
  void (*shadow)(Holder *own) REQUIRES(own->lock); // both-error{{reference to 'own' is ambiguous}} \
                                                   // both-note{{candidate found by name lookup is 'own'}}
  Holder *own; // both-note{{candidate found by name lookup is 'Scoping::own'}}
  void (*shadow_global)(Mutex *global_mu) REQUIRES(global_mu); // both-error{{reference to 'global_mu' is ambiguous}} \
                                                               // both-note{{candidate found by name lookup is 'global_mu'}}
  // It is in scope only for its own attributes.
  void (*cb)(Holder *p) REQUIRES(p->lock);
  int x __attribute__((guarded_by(p->lock))); // both-error{{use of undeclared identifier 'p'}}
};

// guarded_by and pt_guarded_by describe the pointer itself rather than a call
// through it, so a pointee's parameters are not in scope for them and a member
// of the same name is not ambiguous.
struct Guarded {
  void (*cb)(Mutex *own) __attribute__((guarded_by(own)));
  int *(*get)(Mutex *own) __attribute__((pt_guarded_by(own)));
  Mutex own;
  void (*pointee)(Mutex *p) __attribute__((guarded_by(p))); // both-error{{use of undeclared identifier 'p'}}
};

// A nested class's attribute may name a pointee parameter and a member of the
// enclosing class.
struct Outer {
  struct Inner {
    void (*cb)(Holder *h) REQUIRES(h->lock, m);
  };
  static Mutex m;
};

// Template instantiation cannot yet map a pointee's parameters. A name binds
// the same way in a template as elsewhere, but an attribute naming one is
// rejected.
template <typename T>
struct InTemplate {
  void (*cb)(T *h) REQUIRES(h->lock); // both-error{{'requires_capability' attribute in a template cannot name pointee function parameter 'h'}}
  // Ambiguous, as outside a template.
  void (*shadow)(T *own) REQUIRES(own->lock); // both-error{{reference to 'own' is ambiguous}} \
                                              // both-note{{candidate found by name lookup is 'own'}}
  T *own; // both-note{{candidate found by name lookup is 'InTemplate::own'}}
  void method(void (*release)(T) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
              Mutex *mu);
};
InTemplate<Holder> in_template;

template <typename T>
void put_later_template(void (*release)(T) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                        Mutex *mu);
void use_put_later_template(Mutex *mu) { put_later_template<int>(nullptr, mu); }

auto generic_lambda = [](auto x, void (*release)(int) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                         Mutex *mu) {};
void use_generic_lambda(Mutex *mu) { generic_lambda(0, nullptr, mu); }

// An earlier parameter could always be named.
template <typename T>
void put_earlier_template(Mutex *mu, void (*release)(T) RELEASE(mu)) {}
void use_put_earlier_template(Mutex *mu) { put_earlier_template<int>(mu, nullptr); }

// Whether a parameter is declared later depends on its position in the
// prototype, not on how macros spell it.
#define SWAP(a, b) b, a
template <typename T>
void swapped_later(SWAP(Mutex *mu, // early-error{{use of undeclared identifier 'mu'}}
                        void (*release)(T) RELEASE(mu)));
void use_swapped_later(Mutex *mu) { swapped_later<int>(nullptr, mu); }
template <typename T>
void swapped_earlier(SWAP(void (*release)(T) RELEASE(mu), Mutex *mu)) {}
void use_swapped_earlier(Mutex *mu) { swapped_earlier<int>(mu, nullptr); }

// Nor on how deeply its prototype is nested.
template <typename T>
void nested_later(void (*cb)(void (*release)(T) RELEASE(mu), // early-error{{use of undeclared identifier 'mu'}}
                             Mutex *mu));
void use_nested_later(void (*cb)(void (*)(int), Mutex *)) { nested_later<int>(cb); }
template <typename T>
void nested_earlier(Mutex *mu, void (*cb)(void (*release)(T) RELEASE(mu))) {}
void use_nested_earlier(Mutex *mu) { nested_earlier<int>(mu, nullptr); }
