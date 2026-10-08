template <class... T> void ParamPackFunction(T... args);

template <typename T, int U = 1> void function(T x) {}

template <typename A, typename B, typename C, typename D, typename E>
void longFunction(A a, B b, C c, D d, E e) {}

template <> void function<bool, 0>(bool x) {}

/// A Tuple type
///
/// Does Tuple things.
template <typename... Tys> struct tuple {};

/// A function with a tuple parameter
///
/// \param t The input to func_with_tuple_param
tuple<int, int, bool> func_with_tuple_param(tuple<int, int, bool> t) {
  return t;
}
