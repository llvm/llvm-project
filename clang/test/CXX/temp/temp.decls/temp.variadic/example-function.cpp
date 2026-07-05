// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Example function implementation from the variadic templates proposal,
// ISO C++ committee document number N2080.

template<typename Signature> class function;

template<typename R, typename... Args> class invoker_base {
public: 
  virtual ~invoker_base() { } 
  virtual R invoke(Args...) = 0; 
  virtual invoker_base* clone() = 0;
};

template<typename F, typename R, typename... Args> 
class functor_invoker : public invoker_base<R, Args...> {
public: 
  explicit functor_invoker(const F& f) : f(f) { } 
  R invoke(Args... args) { return f(args...); } 
  functor_invoker* clone() { return new functor_invoker(f); }

private:
  F f;
};

template<typename R, typename... Args>
class function<R (Args...)> {
public: 
  typedef R result_type;
  function() : invoker (0) { }
  function(const function& other) : invoker(0) { 
    if (other.invoker)
      invoker = other.invoker->clone();
  }

  template<typename F> function(const F& f) : invoker(0) {
    invoker = new functor_invoker<F, R, Args...>(f);
  }

  ~function() { 
    if (invoker)
      delete invoker;
  }

  function& operator=(const function& other) { 
    function(other).swap(*this); 
    return *this;
  }

  template<typename F> 
  function& operator=(const F& f) {
    function(f).swap(*this); 
    return *this;
  }

  void swap(function& other) { 
    invoker_base<R, Args...>* tmp = invoker; 
    invoker = other.invoker; 
    other.invoker = tmp;
  }

  result_type operator()(Args... args) const { 
    return invoker->invoke(args...);
  }

private: 
  invoker_base<R, Args...>* invoker;
};

template<typename T>
struct add {
  T operator()(T x, T y) { return x + y; }
};

int add_ints(int x, int y) { return x + y; }

void test_function() {
  function<int(int, int)> f2a;
  function<int(int, int)> f2b = add<int>();
  function<int(int, int)> f2c = add<float>();
  function<int(int, int)> f2d(f2b);
  function<int(int, int)> f2e = &add_ints;
  f2c = f2d;
  f2d = &add_ints;
  f2c(1.0, 3);
}
