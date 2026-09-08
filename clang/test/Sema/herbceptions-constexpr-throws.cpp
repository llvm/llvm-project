// RUN: %clang_cc1 -std=c++26 -fherbceptions -fsyntax-only -verify %s
// expected-no-diagnostics

// Constexpr `throws` with `try { } catch throws(std::error e)` block handlers.
// The compiler fabricates a unique opaque domain pointer for
// error_domain<T>::domain() at compile time, so `e == win32_errc::x` and
// `e.code()` work in constant expressions.

namespace std {

struct error_domain_singleton {
  void (*do_cleanup)(unsigned long) noexcept = 0;
  bool (*do_equivalent)(unsigned long, error_domain_singleton const*, unsigned long) noexcept = 0;
  void (*do_name)(unsigned long, int, void*, void*) noexcept = 0;
  void (*do_message)(unsigned long, int, void*, void*) noexcept = 0;
  int (*do_to_errc)(unsigned long) noexcept = 0;
};

class error {
public:
  error() = delete;
  error(error const&) = delete;
  error(error&&) = delete;
  error& operator=(error const&) = delete;
  error& operator=(error&&) = delete;
  constexpr ~error() noexcept {
    auto docleanup{domain_opaque->do_cleanup};
    if (docleanup) docleanup(code_opaque);
  }
  constexpr error_domain_singleton const* domain() const noexcept { return domain_opaque; }
  constexpr __SIZE_TYPE__ code() const noexcept { return code_opaque; }
private:
  error_domain_singleton const* domain_opaque{};
  __SIZE_TYPE__ code_opaque{};
  explicit constexpr error(void const* domain, __SIZE_TYPE__ code) noexcept
      : domain_opaque(static_cast<error_domain_singleton const*>(domain)), code_opaque(code) {}
  friend constexpr error __builtin_herbception_error(void const*, unsigned long);
};

template<typename T>
struct error_domain;

enum class win32_errc : unsigned { success = 0, invalid_function = 1, file_not_found = 2 };

namespace {
constinit error_domain_singleton dummy_domain{};
}

template<>
struct error_domain<win32_errc> {
  using errc_type = win32_errc;
  static constexpr error_domain_singleton const* domain() noexcept { return &dummy_domain; }
  static constexpr __SIZE_TYPE__ code(errc_type e) noexcept {
    return static_cast<unsigned long>(e);
  }
};

template<typename T>
constexpr bool operator==(error const& e, T t) noexcept {
  return error_domain<T>::code(t) == e.code() &&
         error_domain<T>::domain() == e.domain();
}

} // namespace std

constexpr int f(int x) throws {
  if (x == 0) throw throws ::std::win32_errc::file_not_found;
  return 2 * x;
}

constexpr int use_try(int x) {
  try {
    return f(x);
  } catch throws(::std::error e) {
    if (e == ::std::win32_errc::file_not_found)
      return static_cast<int>(e.code());
    return -1;
  }
}

static_assert(use_try(3) == 6, "success");
static_assert(use_try(0) == 2, "failure");
