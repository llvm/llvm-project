// Header for PCH test microsoft-abi-throw-pch.cpp

struct Trivial {};

struct NonTrivial {
  NonTrivial() = default;
  NonTrivial(const NonTrivial &) noexcept {}
  NonTrivial(NonTrivial &&) noexcept {}
};

struct TemplateWithDefault {
  template <typename T>
  static int f() {
    return 0;
  }
  template <typename T = int>
  TemplateWithDefault(TemplateWithDefault &, T = f<T>());
};

inline void throw_trivial() {
  throw Trivial();
}

inline void throw_non_trivial() {
  throw NonTrivial();
}

inline void throw_template(TemplateWithDefault &e) {
  throw e;
}
