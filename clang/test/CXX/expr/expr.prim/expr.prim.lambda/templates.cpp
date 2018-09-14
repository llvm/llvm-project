// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Winvalid-noreturn %s -verify

template<typename T>
void test_attributes() {
  // FIXME: GCC accepts [[gnu::noreturn]] here.
  auto nrl = []() [[gnu::noreturn]] {}; // expected-warning{{attribute 'noreturn' ignored}}
}

template void test_attributes<int>();

template<typename T>
void call_with_zero() {
  [](T *ptr) -> T& { return *ptr; }(0);
}

template void call_with_zero<int>();

template<typename T>
T captures(T x, T y) {
  auto lambda = [=, &y] () -> T {
    T i = x;
    return i + y;
  };

  return lambda();
}

struct X {
  X(const X&);
};

X operator+(X, X);
X operator-(X, X);

template int captures(int, int);
template X captures(X, X);

template<typename T>
int infer_result(T x, T y) {
  auto lambda = [=](bool b) { return x + y; };
  return lambda(true); // expected-error{{no viable conversion from returned value of type 'X' to function return type 'int'}}
}

template int infer_result(int, int);
template int infer_result(X, X); // expected-note{{in instantiation of function template specialization 'infer_result<X>' requested here}}

// Make sure that lambda's operator() can be used from templates.
template<typename F>
void accept_lambda(F f) {
  f(1);
}

template<typename T>
void pass_lambda(T x) {
  accept_lambda([&x](T y) { return x + y; });
}

template void pass_lambda(int);

namespace std {
  class type_info;
}

namespace p2 {
  struct P {
    virtual ~P();
  };

  template<typename T>
  struct Boom {
    Boom(const Boom&) { 
      T* x = 1; // expected-error{{cannot initialize a variable of type 'float *' with an rvalue of type 'int'}}
    }
    void tickle() const;
  };
  
  template<typename R, typename T>
  void odr_used(R &r, Boom<T> boom) {
    const std::type_info &ti
      = typeid([=,&r] () -> R& { // expected-error{{lambda expression in an unevaluated operand}}
          boom.tickle();
          return r; 
        }()); 
  }

  template void odr_used(int&, Boom<int>); // expected-note{{in instantiation of function template specialization}}

  template<typename R, typename T>
  void odr_used2(R &r, Boom<T> boom) {
    const std::type_info &ti
      = typeid([=,&r] () -> R& { // expected-note{{in instantiation of member function 'p2::Boom<float>::Boom' requested here}}
          boom.tickle();
          return r; 
        }()); 
  }

  template void odr_used2(P&, Boom<float>);
}

namespace p5 {
  struct NonConstCopy {
    NonConstCopy(const NonConstCopy&) = delete;
    NonConstCopy(NonConstCopy&);
  };

  template<typename T>
  void double_capture(T &nc) {
    [=] () mutable {
      [=] () mutable {
        T nc2(nc);
      }();
    }();
  }

  template void double_capture(NonConstCopy&);
}

namespace NonLocalLambdaInstantation {
  template<typename T>
  struct X {
    static int value;
  };

  template<typename T>
  int X<T>::value = []{ return T(); }(); // expected-error{{cannot initialize a variable of type 'int' with an rvalue of type 'int *'}}

  template int X<int>::value;
  template int X<float>::value;
  template int X<int*>::value; // expected-note{{in instantiation of static data member }}

  template<typename T>
  void defaults(int x = []{ return T(); }()) { }; // expected-error{{cannot initialize a parameter of type 'int' with an rvalue of type 'int *'}} \
     // expected-note{{passing argument to parameter 'x' here}}

  void call_defaults() {
    defaults<int>();
    defaults<float>();
    defaults<int*>(); // expected-note{{in instantiation of default function argument expression for 'defaults<int *>' required here}}
  }

  template<typename T>
  struct X2 { // expected-note{{in instantiation of default member initializer 'NonLocalLambdaInstantation::X2<int *>::x'}}
    int x = []{ return T(); }(); // expected-error{{cannot initialize a member subobject of type 'int' with an rvalue of type 'int *'}}
  };

  X2<int> x2i;
  X2<float> x2f;
  X2<int*> x2ip; // expected-note {{in evaluation of exception spec}}
}
