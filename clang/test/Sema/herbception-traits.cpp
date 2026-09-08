// RUN: %clang_cc1 -std=c++26 -fherbceptions -fsyntax-only -verify %s
// expected-no-diagnostics

// Herbception type traits: throwsable, is-invoke-fails, and the
// invoke_herbceptions_fails_result value_type/error_type query.

namespace std {
struct error_domain_singleton {};
template <typename T> struct error_domain;

namespace {
constinit error_domain_singleton dummy_domain{};
}

enum class win32_errc : unsigned { success = 0, file_not_found = 2 };
template <> struct error_domain<win32_errc> {
  using errc_type = win32_errc;
  static constexpr error_domain_singleton const *domain() noexcept {
    return &dummy_domain;
  }
  static constexpr unsigned long code(errc_type e) noexcept {
    return static_cast<unsigned long>(e);
  }
};

enum class myerr : unsigned { a = 1 };
template <> struct error_domain<myerr> {
  using errc_type = myerr;
  static constexpr error_domain_singleton const *domain() noexcept {
    return &dummy_domain;
  }
  static constexpr unsigned long code(errc_type e) noexcept {
    return static_cast<unsigned long>(e);
  }
};
} // namespace std

struct Plain {};

int thr(int) throws;
int fls(int) fails{std::myerr};
int plain(int);

// __is_herbceptions_throwsable: a usable error_domain<T> means T can be thrown.
static_assert(__is_herbceptions_throwsable(std::win32_errc), "win32_errc");
static_assert(__is_herbceptions_throwsable(std::myerr), "myerr");
static_assert(!__is_herbceptions_throwsable(Plain), "Plain");
static_assert(!__is_herbceptions_throwsable(int), "int");

// __is_invoke_herbceptions_fails: only `fails{E}` function types, not throws.
static_assert(__is_invoke_herbceptions_fails(decltype(fls)), "fls is fails");
static_assert(!__is_invoke_herbceptions_fails(decltype(thr)), "thr is throws");
static_assert(!__is_invoke_herbceptions_fails(decltype(plain)), "plain");
static_assert(!__is_invoke_herbceptions_fails(int), "int is not a function");
using Fp = int (*)(int) fails{std::myerr};
static_assert(__is_invoke_herbceptions_fails(Fp), "fnptr to fails");

// __invoke_herbceptions_fails_result: {value_type, error_type}.
using R1 = __invoke_herbceptions_fails_result<decltype(fls)>;
static_assert(__is_same(typename R1::value_type, int), "fls returns int");
static_assert(__is_same(typename R1::error_type, std::myerr), "fls fails{myerr}");

using R2 = __invoke_herbceptions_fails_result<decltype(thr)>;
static_assert(__is_same(typename R2::value_type, int), "thr returns int");
static_assert(__is_same(typename R2::error_type, void), "thr has no fails type");

using R3 = __invoke_herbceptions_fails_result<int>;
static_assert(__is_same(typename R3::value_type, void), "int not a function");
static_assert(__is_same(typename R3::error_type, void), "int has no fails type");

using R4 = __invoke_herbceptions_fails_result<Fp>;
static_assert(__is_same(typename R4::value_type, int), "fp returns int");
static_assert(__is_same(typename R4::error_type, std::myerr), "fp fails{myerr}");
