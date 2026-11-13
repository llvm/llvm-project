// RUN: %clang_cc1 %s -std=c++23 -verify -fsyntax-only -Wlifetime-safety -Wlifetime-safety-lifetimebound-violation -Wno-dangling

#include "Inputs/lifetime-analysis.h"

// Explicit object member functions must not be treated as having an implicit
// object argument.
struct Foo {
  template <typename T>
  int get(this Foo &&self, T) {
    return self.field;
  }

  int field;
};

struct Base {};
struct Derived : Base {
  int base(this Base &&, int);
};

template <typename T>
struct Crtp {
  template <typename Self>
  int deduced(this Self &&self, T) {
    return self.field;
  }
};

struct Widget : Crtp<int> {
  int field;
};

struct NoMove {
  NoMove() = default;
  NoMove(NoMove &&) = delete;
  NoMove(const NoMove &) = delete;

  int get(this NoMove &&self, int) {
    return self.field;
  }

  int field;
};

template <typename T>
struct SharedPtr {
  SharedPtr() = default;
  SharedPtr(SharedPtr &&o) : p(o.p) { o.p = nullptr; }
  SharedPtr(const SharedPtr &) = default;

  int get(this SharedPtr &&self, int) {
    return self.p->field;
  }

  T *p = nullptr;
};

void call() {
  Foo().get(0);
  Derived().base(0);
  Widget().deduced(0);
  NoMove().get(0);
  SharedPtr<NoMove>().get(0);
}

struct Holder {
  int field;
  const int *borrow() [[clang::lifetimebound]] { return &field; }
  void consume(this Holder &&self) {}
};

const int *object_arg_is_not_moved(Holder &&h [[clang::lifetimebound]]) {
  static_cast<Holder &&>(h).consume();
  return h.borrow();
}

const int *t1(Holder h) {
  const int *ptr = h.borrow(); // expected-warning {{stack memory associated with parameter 'h' is returned}} \
                               // expected-note {{result of call to 'borrow' aliases the storage of parameter 'h' because the implicit object parameter is marked as lifetimebound}}
  static_cast<Holder &&>(h).consume();
  return ptr; // expected-note {{returned here}}
}

const int *t2(Holder h) {
  const int *ptr = h.borrow(); // expected-warning {{stack memory associated with parameter 'h' is returned}} \
                               // expected-note {{result of call to 'borrow' aliases the storage of parameter 'h' because the implicit object parameter is marked as lifetimebound}}
  std::move(h).consume();
  return ptr; // expected-note {{returned here}}
}

struct Set {
  void insert(this Set &self, const int &x [[clang::lifetime_capture_by(self)]],
              const int &hint);
};

int global;

void capture_by_explicit_object() {
  Set s;
  {
    int local = 0;
    s.insert(local, global); // expected-warning {{local variable 'local' does not live long enough}}
    s.insert(global, local);
  }                          // expected-note {{destroyed here}}
  use(s);                    // expected-note {{later used here}}
}

struct Lookup {
  const int &operator[](this const Lookup &self,
                        const int &key [[clang::lifetimebound]]);
  const int &get(this const Lookup &self,
                 const int &key [[clang::lifetimebound]]);
  const int &self_bound(this const Lookup &self [[clang::lifetimebound]],
                        const int &key);
};

const int &lifetimebound_after_explicit_object_subscript(const Lookup &l) {
  int key = 1;
  return l[key]; // expected-warning {{stack memory associated with local variable 'key' is returned}} \
                 // expected-note {{returned here}}
}

const int &lifetimebound_after_explicit_object_call(const Lookup &l) {
  int key = 1;
  return l.get(key); // expected-warning {{stack memory associated with local variable 'key' is returned}} \
                     // expected-note {{returned here}}
}

const int &lifetimebound_explicit_object(const Lookup &l) {
  int key = 1;
  return l.self_bound(key);
}

const int &dangling_explicit_object() {
  Lookup l;
  return l.self_bound(1); // expected-warning {{stack memory associated with local variable 'l' is returned}} \
                          // expected-note {{returned here}}
}

const int &lifetimebound_after_xvalue_explicit_object(Lookup &l) {
  int key = 1;
  return std::move(l).get(key); // expected-warning {{stack memory associated with local variable 'key' is returned}} \
                                // expected-note {{returned here}}
}

const int &lifetimebound_after_prvalue_explicit_object() {
  int key = 1;
  return Lookup{}.get(key); // expected-warning {{stack memory associated with local variable 'key' is returned}} \
                            // expected-note {{returned here}}
}

const int &dangling_xvalue_explicit_object() {
  Lookup l;
  return std::move(l).self_bound(1); // expected-warning {{stack memory associated with local variable 'l' is returned}} \
                                     // expected-note {{result of call to 'move<Lookup &>' aliases the storage of local variable 'l'}} \
                                     // expected-note {{returned here}}
}

const int &dangling_prvalue_explicit_object() {
  return Lookup{}.self_bound(1); // expected-warning {{stack memory associated with temporary object is returned}} \
                                 // expected-note {{returned here}}
}

const int &xvalue_explicit_object_param(Lookup &l) {
  return std::move(l).self_bound(1);
}

struct Deduced {
  template <typename Self>
  const int &get(this Self &&self [[clang::lifetimebound]]);
  const int &get_auto(this auto &&self [[clang::lifetimebound]]);
};

const int &dangling_deduced_explicit_object() {
  Deduced d;
  return d.get(); // expected-warning {{stack memory associated with local variable 'd' is returned}} \
                  // expected-note {{returned here}}
}

const int &dangling_abbreviated_explicit_object() {
  Deduced d;
  return d.get_auto(); // expected-warning {{stack memory associated with local variable 'd' is returned}} \
                       // expected-note {{returned here}}
}

const int &dangling_prvalue_deduced_explicit_object() {
  return Deduced{}.get(); // expected-warning {{stack memory associated with temporary object is returned}} \
                          // expected-note {{returned here}}
}

const int &deduced_explicit_object_param(const Deduced &d) { return d.get(); }

// The object-argument heuristics apply to an explicit object parameter too.
struct [[gsl::Owner]] Buffer {};
struct [[gsl::Pointer]] BufferView {
  BufferView(const Buffer &b [[clang::lifetimebound]]);
  int *get(this const BufferView &self);
  int &operator*(this const BufferView &self);
};
struct [[gsl::Pointer]] ValueView {
  ValueView(const Buffer &b [[clang::lifetimebound]]);
  int *get(this ValueView self);
};
struct [[gsl::Pointer]] TextView {
  const char *p;
};
struct [[gsl::Owner]] Text {
  operator TextView(this const Text &self);
};
struct Assignable {
  Assignable &operator=(this Assignable &self, const Assignable &other);
};

int *get_through_explicit_object() {
  Buffer b;
  BufferView w(b); // expected-warning {{stack memory associated with local variable 'b' is returned}}
  return w.get();  // expected-note {{returned here}}
}

int *deref_through_explicit_object() {
  Buffer b;
  BufferView w(b); // expected-warning {{stack memory associated with local variable 'b' is returned}}
  return &*w;      // expected-note {{result of call to 'operator*' aliases the storage of local variable 'b'}} \
                   // expected-note {{returned here}}
}

int *get_through_by_value_explicit_object() {
  Buffer b;
  ValueView w(b); // expected-warning {{stack memory associated with local variable 'b' is returned}}
  return w.get(); // expected-note {{returned here}}
}

TextView convert_through_explicit_object() {
  Text t;
  return t; // expected-warning {{stack memory associated with local variable 't' is returned}} \
            // expected-note {{returned here}}
}

Assignable &assign_through_explicit_object() {
  Assignable a;
  return a = Assignable(); // expected-warning {{stack memory associated with local variable 'a' is returned}} \
                           // expected-note {{returned here}}
}

int *get_through_explicit_object_ok(const Buffer &b) {
  BufferView w(b);
  return w.get();
}

struct Sink {
  void take(this Sink &self, std::string &&s);
};

std::string_view moved_after_explicit_object(Sink &sink) {
  std::string s;
  std::string_view v = s;  // expected-warning {{stack memory associated with local variable 's' may be returned later}}
  sink.take(std::move(s)); // expected-note {{potentially moved here}}
  return v;                // expected-note {{returned here}}
}
