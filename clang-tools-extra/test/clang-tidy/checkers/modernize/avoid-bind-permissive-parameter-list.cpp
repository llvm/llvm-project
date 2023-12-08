// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-avoid-bind %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     modernize-avoid-bind.PermissiveParameterList: true}}" --

namespace std {
inline namespace impl {
template <class Fp, class... Arguments>
class bind_rt {};

template <class Fp, class... Arguments>
bind_rt<Fp, Arguments...> bind(Fp &&, Arguments &&...);
} // namespace impl

template <typename T>
T ref(T &t);
} // namespace std

int add(int x, int y) { return x + y; }

// Let's fake a minimal std::function-like facility.
namespace std {
template <typename _Tp>
_Tp declval();

template <typename _Functor, typename... _ArgTypes>
struct __res {
  template <typename... _Args>
  static decltype(declval<_Functor>()(_Args()...)) _S_test(int);

  template <typename...>
  static void _S_test(...);

  using type = decltype(_S_test<_ArgTypes...>(0));
};

template <typename>
struct function;

template <typename... _ArgTypes>
struct function<void(_ArgTypes...)> {
  template <typename _Functor,
            typename = typename __res<_Functor, _ArgTypes...>::type>
  function(_Functor) {}
};
} // namespace std

struct placeholder {};
placeholder _1;

void testLiteralParameters() {
  auto AAA = std::bind(add, 2, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind [modernize-avoid-bind]
  // CHECK-FIXES: auto AAA = [](auto && ...) { return add(2, 2); };

  auto BBB = std::bind(add, _1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind [modernize-avoid-bind]
  // CHECK-FIXES: auto BBB = [](auto && PH1, auto && ...) { return add(std::forward<decltype(PH1)>(PH1), 2); };
}
