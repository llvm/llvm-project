#include <type_traits>

namespace absl {

template <typename... Ts>
struct conjunction : std::true_type {};

template <typename T, typename... Ts>
struct conjunction<T, Ts...>
    : std::conditional<T::value, conjunction<Ts...>, T>::type {};

template <typename T>
struct conjunction<T> : T {};

template <typename... Ts>
struct disjunction : std::false_type {};

template <typename T, typename... Ts>
struct disjunction<T, Ts...>
    : std::conditional<T::value, T, disjunction<Ts...>>::type {};

template <typename T>
struct disjunction<T> : T {};

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;


template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

using std::in_place;
using std::in_place_t;
} // namespace absl
