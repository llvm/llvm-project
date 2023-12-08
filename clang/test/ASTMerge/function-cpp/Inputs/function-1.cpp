
template<typename T> constexpr T add(T arg1, T arg2) {
  return arg1 + arg2;
}

template<> constexpr int add(int arg1, int arg2) {
  return arg1 + arg2 + 2;
}
