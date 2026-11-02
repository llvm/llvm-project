// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

// Reduced from libstdc++'s std::sort(..., std::ref(std::function<...>)), which
// was incorrectly rejected with "exception specification ... uses itself".

template <class T> T &&declval() noexcept;
template <class T> T &&move(T &t) noexcept { return static_cast<T &&>(t); }

// std::reference_wrapper
template <class T> class reference_wrapper {
  static void fun(T &) noexcept;
  static void fun(T &&) = delete;

public:
  template <class U, class = decltype(fun(declval<U>()))>
  reference_wrapper(U &&);
};

template <class T> reference_wrapper<T> ref(T &t) {
  return reference_wrapper<T>(t);
}

// std::function
template <class Signature> class function {
public:
  function();
  template <class F>
  function(F &&) noexcept(__is_nothrow_constructible(__remove_cvref(F), F));
};

// __gnu_cxx::__ops::_Iter_comp_iter
template <class Compare> struct iter_comp_iter {
  explicit iter_comp_iter(Compare);
};

// __gnu_cxx::__ops::_Iter_comp_val
template <class Compare> struct iter_comp_val {
  explicit iter_comp_val(Compare);
  explicit iter_comp_val(const iter_comp_iter<Compare> &);
  explicit iter_comp_val(iter_comp_iter<Compare> &&);
};

// __gnu_cxx::__ops::__iter_comp_val
template <class Compare>
iter_comp_val<Compare> make_iter_comp_val(iter_comp_iter<Compare>);

// std::__adjust_heap
template <class Compare> void adjust_heap(iter_comp_iter<Compare> comp) {
  decltype(make_iter_comp_val(move(comp))) cmp(move(comp));
}

void test() {
  function<bool(const int &, const int &)> comp;
  adjust_heap(iter_comp_iter(ref(comp)));
}
