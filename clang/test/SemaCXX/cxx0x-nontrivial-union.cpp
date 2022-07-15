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
    disable_dtor(U &&...u) : val(forward<U>(u)...) {}
    ~disable_dtor() {}
  };

  struct deleted_dtor {
    deleted_dtor(int n, char c) : n(n), c(c) {}
    int n;
    char c;
    ~deleted_dtor() = delete;
  };

  disable_dtor<deleted_dtor> dd(4, 'x');
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
      struct {  // expected-note {{default constructor of 'Test2<pr16061::X>' is implicitly deleted because variant field '' has a non-trivial default constructor}}
        T x;
      };
    };
  };

  Test2<X> t2x;  // expected-error {{call to implicitly-deleted default constructor of 'Test2<pr16061::X>'}}
}
