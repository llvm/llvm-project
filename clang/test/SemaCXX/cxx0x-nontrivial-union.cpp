// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct non_trivial {
  non_trivial();
  non_trivial(const non_trivial&);
  non_trivial& operator = (const non_trivial&);
  ~non_trivial();
};

union u {
  non_trivial nt;
};
union u2 {
  non_trivial nt;
  int k;
  u2(int k) : k(k) {}
  u2() : nt() {}
};

union static_data_member {
  static int i;
};
int static_data_member::i;

union bad {
  int &i; // expected-error {{union member 'i' has reference type 'int &'}}
};

struct s {
  union {
    non_trivial nt;
  };
};

// Don't crash on this.
struct TemplateCtor { template<typename T> TemplateCtor(T); };
union TemplateCtorMember { TemplateCtor s; };

template<typename T> struct remove_ref { typedef T type; };
template<typename T> struct remove_ref<T&> { typedef T type; };
template<typename T> struct remove_ref<T&&> { typedef T type; };
template<typename T> T &&forward(typename remove_ref<T>::type &&t);
template<typename T> T &&forward(typename remove_ref<T>::type &t);
template<typename T> typename remove_ref<T>::type &&move(T &&t);

using size_t = decltype(sizeof(int));
void *operator new(size_t, void *p) noexcept { return p; }

namespace disabled_dtor {
  template<typename T>
  union disable_dtor {
    T val;
    template<typename...U>
    disable_dtor(U &&...u) : val(forward<U>(u)...) {} // expected-error {{attempt to use a deleted function}}
    ~disable_dtor() {}
  };

  struct deleted_dtor {
    deleted_dtor(int n, char c) : n(n), c(c) {}
    int n;
    char c;
    ~deleted_dtor() = delete; // expected-note {{'~deleted_dtor' has been explicitly marked deleted here}}
  };

  disable_dtor<deleted_dtor> dd(4, 'x');  // expected-note {{in instantiation of function template specialization 'disabled_dtor::disable_dtor<disabled_dtor::deleted_dtor>::disable_dtor<int, char>' requested here}}
}

namespace optional {
  template<typename T> struct optional {
    bool has;
    union { T value; };

    optional() : has(false) {}
    template<typename...U>
    optional(U &&...u) : has(true), value(forward<U>(u)...) {}

    optional(const optional &o) : has(o.has) {
      if (has) new (&value) T(o.value);
    }
    optional(optional &&o) : has(o.has) {
      if (has) new (&value) T(move(o.value));
    }

    optional &operator=(const optional &o) {
      if (has) {
        if (o.has)
          value = o.value;
        else
          value.~T();
      } else if (o.has) {
        new (&value) T(o.value);
      }
      has = o.has;
    }
    optional &operator=(optional &&o) {
      if (has) {
        if (o.has)
          value = move(o.value);
        else
          value.~T();
      } else if (o.has) {
        new (&value) T(move(o.value));
      }
      has = o.has;
    }

    ~optional() {
      if (has)
        value.~T();
    }

    explicit operator bool() const { return has; }
    T &operator*() { return value; }
  };

  optional<non_trivial> o1;
  optional<non_trivial> o2{non_trivial()};
  optional<non_trivial> o3{*o2};
  void f() {
    if (o2)
      o1 = o2;
    o2 = optional<non_trivial>();
  }
}

namespace pr16061 {
  struct X { X(); };

  template<typename T> struct Test1 {
    union {
      struct {
        X x;
      };
    };
  };

  template<typename T> struct Test2 {
    union {
      struct {  // expected-note-re {{default constructor of 'Test2<pr16061::X>' is implicitly deleted because variant field 'struct (anonymous struct at{{.+}})' has a non-trivial default constructor}}
        T x;
      };
    };
  };

  Test2<X> t2x;  // expected-error {{call to implicitly-deleted default constructor of 'Test2<X>'}}
}

namespace GH48416 {

struct non_trivial_constructor {
    constexpr non_trivial_constructor() : x(100) {}
    int x;
};


union U1 {
    int a;
    non_trivial_constructor b; // expected-note {{has a non-trivial default constructor}}
};

union U2 {
    int a{1000};
    non_trivial_constructor b;
};

union U3 {
    int a;
    non_trivial_constructor b{};
};

union U4 {
    int a{}; // expected-note {{previous initialization is here}}
    non_trivial_constructor b{}; // expected-error {{initializing multiple members of union}}
};

U1 u1; // expected-error {{call to implicitly-deleted default constructor}}
U2 u2;
U3 u3;
U4 u4;

static_assert(U2().a == 1000, "");
static_assert(U3().a == 1000, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@-2 {{read of member 'a' of union with active member 'b'}}
static_assert(U2().b.x == 100, "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@-2 {{read of member 'b' of union with active member 'a'}}
static_assert(U3().b.x == 100, "");

} // namespace GH48416

namespace GH81774 {
struct Handle {
    Handle(int) {}
};
// Should be well-formed because NoState has a brace-or-equal-initializer.
union a {
        int NoState = 0;
        Handle CustomState;
} b;
} // namespace GH81774
