// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s
template<typename T, typename U>
struct X0 {
  void f(T x, U y) { 
    (void)(x + y); // expected-error{{invalid operands}}
  }
};

struct X1 { };

template struct X0<int, float>;
template struct X0<int*, int>;
template struct X0<int X1::*, int>; // expected-note{{instantiation of}}

template<typename T>
struct X2 {
  void f(T);

  T g(T x, T y) {
    /* DeclStmt */;
    T *xp = &x, &yr = y; // expected-error{{pointer to a reference}}
    /* NullStmt */;
  }
};

template struct X2<int>;
template struct X2<int&>; // expected-note{{instantiation of}}

template<typename T>
struct X3 {
  void f(T) {
    Label:
    T x;
    goto Label;
  }
};

template struct X3<int>;

template <typename T> struct X4 {
  T f() const {
    return; // expected-error{{non-void function 'f' should return a value}}
  }
  
  T g() const {
    return 1; // expected-error{{void function 'g' should not return a value}}
  }
};

template struct X4<void>; // expected-note{{in instantiation of}}
template struct X4<int>; // expected-note{{in instantiation of}}

struct Incomplete; // expected-note 2{{forward declaration}}

template<typename T> struct X5 {
  T f() { } // expected-error{{incomplete result type}}
};
void test_X5(X5<Incomplete> x5); // okay!

template struct X5<Incomplete>; // expected-note{{instantiation}}

template<typename T, typename U, typename V> struct X6 {
  U f(T t, U u, V v) {
    // IfStmt
    if (t > 0)
      return u;
    else { 
      if (t < 0)
        return v; // expected-error{{cannot initialize return object of type}}
    }

    if (T x = t) {
      t = x;
    }
    return v; // expected-error{{cannot initialize return object of type}}
  }
};

struct ConvertibleToInt {
  operator int() const;
};

template struct X6<ConvertibleToInt, float, char>;
template struct X6<bool, int, int*>; // expected-note{{instantiation}}

template <typename T> struct X7 {
  void f() {
    void *v = this;
  }
};

template struct X7<int>;

template<typename T> struct While0 {
  void f(T t) {
    while (t) {
    }

    while (T t2 = T()) ;
  }
};

template struct While0<float>;

template<typename T> struct Do0 {
  void f(T t) {
    do {
    } while (t); // expected-error{{not contextually}}
  }
};

struct NotConvertibleToBool { };
template struct Do0<ConvertibleToInt>;
template struct Do0<NotConvertibleToBool>; // expected-note{{instantiation}}

template<typename T> struct For0 {
  void f(T f, T l) {
    for (; f != l; ++f) {
      if (*f)
        continue;
      else if (*f == 17)
        break;
    }
  }
};

template struct For0<int*>;

template<typename T> struct Member0 {
  void f(T t) {
    t;
    t.f;
    t->f;
    
    T* tp;
    tp.f; // expected-error{{member reference base type 'T *' is not a structure or union}}
    tp->f;

    this->f;
    this.f; // expected-error{{member reference base type 'Member0<T> *' is not a structure or union}}
  }
};

template<typename T, typename U> struct Switch0 {
  U f(T value, U v0, U v1, U v2) {
    switch (value) {
    case 0: return v0;

    case 1: return v1;

    case 2: // fall through

    default:
      return  v2;
    }
  }
};

template struct Switch0<int, float>;

template<typename T, int I1, int I2> struct Switch1 {
  T f(T x, T y, T z) {
    switch (x) {
    case I1: return y; // expected-note{{previous}}
    case I2: return z; // expected-error{{duplicate}}
    default: return x;
    }
  }
};

template struct Switch1<int, 1, 2>;
template struct Switch1<int, 2, 2>; // expected-note{{instantiation}}

template<typename T> struct IndirectGoto0 {
  void f(T x) {
    // FIXME: crummy error message below
    goto *x; // expected-error{{incompatible}}

  prior:
    T prior_label;
    prior_label = &&prior; // expected-error{{assigning to 'int'}}

    T later_label;
    later_label = &&later; // expected-error{{assigning to 'int'}}

  later:
    (void)(1+1);
  }
};

template struct IndirectGoto0<void*>;
template struct IndirectGoto0<int>; // expected-note{{instantiation}}

template<typename T> struct TryCatch0 {
  void f() {
    try {
    } catch (T t) { // expected-error{{incomplete type}} \
                    // expected-error{{abstract class}}
    } catch (...) {
    }
  }
};

struct Abstract {
  virtual void foo() = 0; // expected-note{{pure virtual}}
};

template struct TryCatch0<int>; // okay
template struct TryCatch0<Incomplete*>; // expected-note{{instantiation}}
template struct TryCatch0<Abstract>; // expected-note{{instantiation}}

// PR4383
template<typename T> struct X;
template<typename T> struct Y : public X<T> {
  Y& x() { return *this; }
};

// Make sure our assertions don't get too uppity.
namespace test0 {
  template <class T> class A { void foo(T array[10]); };
  template class A<int>;
}

namespace PR7016 {
  template<typename T> void f() { T x = x; }
  template void f<int>();
}

namespace PR9880 {
  struct lua_State;
  struct no_tag { char a; };			// (A)
  struct yes_tag { long a; long b; };	// (A)

  template <typename T>
  struct HasIndexMetamethod {
    template <typename U>
    static no_tag check(...);
    template <typename U>
    static yes_tag check(char[sizeof(&U::luaIndex)]);
    enum { value = sizeof(check<T>(0)) == sizeof(yes_tag) };
  };
  
  class SomeClass {
  public:
    int luaIndex(lua_State* L);
  };
  
  int i = HasIndexMetamethod<SomeClass>::value;
}
