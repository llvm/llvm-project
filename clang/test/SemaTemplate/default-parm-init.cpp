// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

namespace std {

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

}

template<typename TemplateParam>
struct Problem {
  template<typename FunctionTemplateParam>
  constexpr int FuncAlign(int param = alignof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof(int param = sizeof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncAlign2(int param = alignof(TemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof2(int param = sizeof(TemplateParam));
};

template<typename TemplateParam>
struct Problem<TemplateParam*> {
  template<typename FunctionTemplateParam>
  constexpr int FuncAlign(int param = alignof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof(int param = sizeof(FunctionTemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncAlign2(int param = alignof(TemplateParam));

  template<typename FunctionTemplateParam>
  constexpr int FuncSizeof2(int param = sizeof(TemplateParam));
};

template<typename TemplateParam>
template<typename FunctionTemplateParam>
constexpr int Problem<TemplateParam*>::FuncAlign(int param) {
	return 2U*param;
}

template<typename TemplateParam>
template<typename FunctionTemplateParam>
constexpr int Problem<TemplateParam*>::FuncSizeof(int param) {
    return 2U*param;
}

template<typename TemplateParam>
template<typename FunctionTemplateParam>
constexpr int Problem<TemplateParam*>::FuncAlign2(int param) {
	return 2U*param;
}

template<typename TemplateParam>
template<typename FunctionTemplateParam>
constexpr int Problem<TemplateParam*>::FuncSizeof2(int param) {
	return 2U*param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncAlign(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncSizeof(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncAlign2(int param) {
	return param;
}

template <>
template<typename FunctionTemplateParam>
constexpr int Problem<int>::FuncSizeof2(int param) {
	return param;
}

void foo() {
    Problem<int> p = {};
    static_assert(p.FuncAlign<char>() == alignof(char));
    static_assert(p.FuncSizeof<char>() == sizeof(char));
    static_assert(p.FuncAlign2<char>() == alignof(int));
    static_assert(p.FuncSizeof2<char>() == sizeof(int));
    Problem<short*> q = {};
    static_assert(q.FuncAlign<char>() == 2U * alignof(char));
    static_assert(q.FuncSizeof<char>() == 2U * sizeof(char));
    static_assert(q.FuncAlign2<char>() == 2U *alignof(short));
    static_assert(q.FuncSizeof2<char>() == 2U * sizeof(short));
}

template <typename T>
class A {
 public:
  void run(
    std::function<void(T&)> f1 = [](auto&&) {},
    std::function<void(T&)> f2 = [](auto&&) {});
 private:
  class Helper {
   public:
    explicit Helper(std::function<void(T&)> f2) : f2_(f2) {}
    std::function<void(T&)> f2_;
  };
};

template <typename T>
void A<T>::run(std::function<void(T&)> f1,
               std::function<void(T&)> f2) {
  Helper h(f2);
}

struct B {};

int main() {
    A<B> a;
    a.run([&](auto& l) {});
    return 0;
}
