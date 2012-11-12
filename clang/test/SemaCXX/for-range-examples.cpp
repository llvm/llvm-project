// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

namespace value_range_detail {
  template<typename T>
  class value_range_iter {
    T t;
  public:
    value_range_iter(const T &t) : t(t) {}
    T operator*() const { return t; }
    bool operator!=(const value_range_iter &o) const { return t != o.t; }
    value_range_iter &operator++() { ++t; return *this; }
  };

  template<typename T>
  struct value_range {
    value_range(const T &a, const T &b) : begin_(a), end_(b) {}
    value_range_iter<T> begin_, end_;
  };

  template<typename T>
  value_range_iter<T> begin(const value_range<T> &r) { return r.begin_; }
  template<typename T>
  value_range_iter<T> end(const value_range<T> &r) { return r.end_; }


  struct end_t {};

  template<typename T>
  class value_range_step_iter {
    T it, step;
  public:
    value_range_step_iter(const T &it, const T &step) : it(it), step(step) {}
    T operator*() const { return it; }
    bool operator!=(value_range_step_iter end) const { return it != end.it; }
    value_range_step_iter &operator++() { it += step; return *this; }
  };

  template<typename T>
  class value_range_step {
    T it, step, end_;
  public:
    value_range_step(const T &it, const T &end, const T &step) :
      it(it), end_(end), step(step) {}
    typedef value_range_step_iter<T> iterator;
    iterator begin() const { return iterator(it, step); }
    iterator end() const { return iterator(end_, step); }
  };
}

template<typename T>
value_range_detail::value_range<T> range(const T &a, const T &b) { return value_range_detail::value_range<T>(a, b); }

template<typename T>
value_range_detail::value_range_step<T> range(const T &a, const T &b, const T &step) { return value_range_detail::value_range_step<T>(a, b, step); }


namespace map_range {
  template<typename T>
  class vector {
    T storage[100];
    decltype(sizeof(char)) size;
  public:
    vector() : size() {}
    void push_back(T t) { storage[size++] = t; }
    T *begin() { return storage; }
    T *end() { return storage + size; }
  };

  template<typename T> struct tuple_elem {
    T t;
    tuple_elem() {}
    tuple_elem(T t) : t(t) {}
  };
  template<typename... A>
  struct tuple : tuple_elem<A>... {
    tuple() : tuple_elem<A>()... {}
    tuple(A... a) : tuple_elem<A>(a)... {}
    template<typename B> B &get() { return tuple_elem<B>::t; }
  };

  template<typename F, typename I>
  class map_iter {
    F f;
    I i;
  public:
    map_iter(F f, I i) : f(f), i(i) {}
    auto operator*() const -> decltype(f(*i)) { return f(*i); }
    bool operator!=(const map_iter &o) const { return i != o.i; }
    map_iter &operator++() { ++i; return *this; }
  };

  template<typename T>
  struct iter_pair {
    T begin_, end_;
    iter_pair(T begin, T end) : begin_(begin), end_(end) {}
  };
  template<typename T> T begin(iter_pair<T> p) { return p.begin_; }
  template<typename T> T end(iter_pair<T> p) { return p.end_; }

  template<typename...> class mem_fun_impl;
  template<typename R, typename T, typename... A>
  class mem_fun_impl<R (T::*)(A...)> {
    typedef R (T::*F)(A...);
    F f;
  public:
    mem_fun_impl(F f) : f(f) {}
    R operator()(T &t, A &&...a) const { return (t.*f)(static_cast<A&&>(a)...); }
  };
  template<typename F> mem_fun_impl<F> mem_fun(F f) { return mem_fun_impl<F>(f); }

  template<typename F, typename T>
  auto map(const F &f, T &t) -> iter_pair<map_iter<F, decltype(t.begin())>> {
    typedef map_iter<F, decltype(t.begin())> iter;
    return iter_pair<iter>(iter(f, t.begin()), iter(f, t.end()));
  }
}

#define assert(b) if (!(b)) { return 1; }
int main() {
  int total = 0;

  for (auto n : range(1, 5)) {
    total += n;
  }
  assert(total == 10);

  for (auto n : range(10, 100, 10)) {
    total += n;
  }
  assert(total == 460);

  map_range::vector<char> chars;
  chars.push_back('a');
  chars.push_back('b');
  chars.push_back('c');
  for (char c : chars) {
    ++total;
  }
  assert(total == 463);

  typedef map_range::tuple<int, double> T;
  map_range::vector<T> pairs;
  pairs.push_back(T(42, 12.9));
  pairs.push_back(T(6, 4.2));
  pairs.push_back(T(9, 1.1));
  for (auto a : map(map_range::mem_fun(&T::get<int>), pairs)) {
    total += a;
  }
  assert(total == 500);
}

// PR11793
namespace test2 {
  class A {
    int xs[10]; // expected-note {{implicitly declared private here}}
  };
  void test(A &a) {
    for (int x : a.xs) { } // expected-error {{'xs' is a private member of 'test2::A'}}
  }
}

namespace test3 {
  // Make sure this doesn't crash
  struct A {};
  struct B { ~B(); operator bool(); };
  struct C { B operator!=(const C&); C& operator++(); int operator*(); };
  C begin(const A&);
  C end(const A&);
  template<typename T> void f() { for (auto a : A()) {} }
  void g() { f<int>(); }
}

namespace test4 {
  void f() {
    int y;

    // Make sure these don't crash. Better diagnostics would be nice.
    for (: {1, 2, 3}) {} // expected-error {{expected expression}} expected-error {{expected ';'}}
    for (x : {1, 2, 3}) {} // expected-error {{undeclared identifier}} expected-error {{expected ';'}}
    for (y : {1, 2, 3}) {} // expected-error {{must declare a variable}} expected-warning {{result unused}}
  }
}
