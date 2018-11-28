// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s

// Don't crash if the argument to __builtin_constant_p isn't scalar.
template <typename T>
constexpr bool is_constant(const T v) {
  return __builtin_constant_p(v);
}

template <typename T>
class numeric {
 public:
  using type = T;

  template <typename S>
  constexpr numeric(S value)
      : value_(static_cast<T>(value)) {}

 private:
  const T value_;
};

bool bcp() {
  return is_constant(numeric<int>(1));
}
