// RUN: %clang_cc1 -std=c++11 %s -fsyntax-only -fcxx-exceptions

template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };
template<typename T> struct remove_reference<T&&> { typedef T type; };

template<typename T> constexpr T &&forward(typename remove_reference<T>::type &t) noexcept { return static_cast<T&&>(t); }
template<typename T> constexpr T &&forward(typename remove_reference<T>::type &&t) noexcept { return static_cast<T&&>(t); }
template<typename T> constexpr typename remove_reference<T>::type &&move(T &&t) noexcept { return static_cast<typename remove_reference<T>::type&&>(t); }

template<typename T> T declval() noexcept;

namespace detail {
  template<unsigned N> struct select {}; // : integral_constant<unsigned, N> {};
  template<typename T> struct type {};

  template<typename...T> union either_impl;

  template<> union either_impl<> {
    void get(...);
    void destroy(...) { throw "logic_error"; }
  };

  template<typename T, typename...Ts> union either_impl<T, Ts...> {
  private:
    T val;
    either_impl<Ts...> rest;
    typedef either_impl<Ts...> rest_t;

  public:
    constexpr either_impl(select<0>, T &&t) : val(move(t)) {}

    template<unsigned N, typename U>
    constexpr either_impl(select<N>, U &&u) : rest(select<N-1>(), move(u)) {}

    constexpr static unsigned index(type<T>) { return 0; }
    template<typename U>
    constexpr static unsigned index(type<U> t) {
      return decltype(rest)::index(t) + 1;
    }

    void destroy(unsigned elem) {
      if (elem)
        rest.destroy(elem - 1);
      else
        val.~T();
    }

    constexpr const T &get(select<0>) { return val; }
    template<unsigned N> constexpr const decltype(static_cast<const rest_t&>(rest).get(select<N-1>{})) get(select<N>) {
      return rest.get(select<N-1>{});
    }
  };
}

template<typename T>
struct a {
  T value;
  template<typename...U>
  constexpr a(U &&...u) : value{forward<U>(u)...} {}
};
template<typename T> using an = a<T>;

template<typename T, typename U> T throw_(const U &u) { throw u; }

template<typename...T>
class either {
  unsigned elem;
  detail::either_impl<T...> impl;
  typedef decltype(impl) impl_t;

public:
  template<typename U>
  constexpr either(a<U> &&t) :
    elem(impl_t::index(detail::type<U>())),
    impl(detail::select<impl_t::index(detail::type<U>())>(), move(t.value)) {}

  // Destruction disabled to allow use in a constant expression.
  // FIXME: declare a destructor iff any element has a nontrivial destructor
  //~either() { impl.destroy(elem); }

  constexpr unsigned index() noexcept { return elem; }

  template<unsigned N> using const_get_result =
    decltype(static_cast<const impl_t&>(impl).get(detail::select<N>{}));

  template<unsigned N>
  constexpr const_get_result<N> get() {
    // Can't just use throw here, since that makes the conditional a prvalue,
    // which means we return a reference to a temporary.
    return (elem != N ? throw_<const_get_result<N>>("bad_either_get")
                      : impl.get(detail::select<N>{}));
  }

  template<typename U>
  constexpr const U &get() {
    return get<impl_t::index(detail::type<U>())>();
  }
};

typedef either<int, char, double> icd;
constexpr icd icd1 = an<int>(4);
constexpr icd icd2 = a<char>('x');
constexpr icd icd3 = a<double>(6.5);

static_assert(icd1.get<int>() == 4, "");
static_assert(icd2.get<char>() == 'x', "");
static_assert(icd3.get<double>() == 6.5, "");

struct non_triv {
  constexpr non_triv() : n(5) {}
  int n;
};
constexpr either<const icd*, non_triv> icd4 = a<const icd*>(&icd2);
constexpr either<const icd*, non_triv> icd5 = a<non_triv>();

static_assert(icd4.get<const icd*>()->get<char>() == 'x', "");
static_assert(icd5.get<non_triv>().n == 5, "");
