// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fsyntax-only %s
//
// [[noreturn]] with herbceptions:
// - A [[noreturn]] function that calls another [[noreturn]] function is valid
// - The herbceptions error channel does not affect [[noreturn]] semantics

namespace std {
class error {
public:
  error() = delete;
  error(error const &) = delete;
  error &operator=(error const &) = delete;
  constexpr ~error() noexcept {}
  __SIZE_TYPE__ code() const noexcept { return code_opaque; }

private:
  void const *domain_opaque{};
  __SIZE_TYPE__ code_opaque{};
  explicit constexpr error(void const *domain, __SIZE_TYPE__ code) noexcept
      : domain_opaque(domain), code_opaque(code) {}
};
} // namespace std

// Forward declaration of a [[noreturn]] function
[[noreturn]] void noreturn_throws() throws;

// A function that conditionally calls a [[noreturn]] function
int bar(int val) throws
{
  if (val == 0)
    return 5;
  noreturn_throws();
}

// Definition of the [[noreturn]] function
[[noreturn]] void noreturn_throws() throws
{
  while (true) {}
}

// [[noreturn]] function with return_failure
[[noreturn]] void noreturn_fails() return_failure{int};

int baz(int val) return_failure{int}
{
  if (val == 0)
    return 5;
  noreturn_fails();
}
