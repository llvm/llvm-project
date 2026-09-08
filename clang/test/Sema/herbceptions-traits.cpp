// RUN: %clang_cc1 -std=c++26 -fherbceptions -fsyntax-only -verify %s
// expected-no-diagnostics

// Herbception type traits: throwsable, is-invoke-fails, the
// invoke_herbceptions_fails_result value_type/error_type query,
// plus the throws-constructible, throws-copy/move-constructible,
// and throws-invocable traits.

// Provide a minimal error_domain specialization so that
// __is_herbceptions_throwsable can be tested.
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
  static constexpr __SIZE_TYPE__ code(errc_type e) noexcept {
    return static_cast<unsigned long>(e);
  }
};

enum class myerr : unsigned { a = 1 };
template <> struct error_domain<myerr> {
  using errc_type = myerr;
  static constexpr error_domain_singleton const *domain() noexcept {
    return &dummy_domain;
  }
  static constexpr __SIZE_TYPE__ code(errc_type e) noexcept {
    return static_cast<unsigned long>(e);
  }
};
} // namespace std

struct Plain {};

int thr(int) throws;
int fls(int) return_failure{std::myerr};
int plain(int);

// __is_herbceptions_throwsable: a usable error_domain<T> means T can be thrown.
static_assert(__is_herbceptions_throwsable(std::win32_errc), "win32_errc");
static_assert(__is_herbceptions_throwsable(std::myerr), "myerr");
static_assert(!__is_herbceptions_throwsable(Plain), "Plain");
static_assert(!__is_herbceptions_throwsable(int), "int");

// __is_invoke_herbceptions_return_failure: only `return_failure{E}` function types, not throws.
static_assert(__is_invoke_herbceptions_return_failure(decltype(fls)), "fls is return_failure");
static_assert(!__is_invoke_herbceptions_return_failure(decltype(thr)), "thr is throws");
static_assert(!__is_invoke_herbceptions_return_failure(decltype(plain)), "plain");
static_assert(!__is_invoke_herbceptions_return_failure(int), "int is not a function");
using Fp = int (*)(int) return_failure{std::myerr};
static_assert(__is_invoke_herbceptions_return_failure(Fp), "fnptr to return_failure");

// __invoke_herbceptions_return_failure_result: {value_type, error_type}.
using R1 = __invoke_herbceptions_return_failure_result<decltype(fls)>;
static_assert(__is_same(typename R1::value_type, int), "fls returns int");
static_assert(__is_same(typename R1::error_type, std::myerr), "fls return_failure{myerr}");

using R2 = __invoke_herbceptions_return_failure_result<decltype(thr)>;
static_assert(__is_same(typename R2::value_type, int), "thr returns int");
static_assert(__is_same(typename R2::error_type, void), "thr has no return_failure type");

using R3 = __invoke_herbceptions_return_failure_result<int>;
static_assert(__is_same(typename R3::value_type, void), "int not a function");
static_assert(__is_same(typename R3::error_type, void), "int has no return_failure type");

using R4 = __invoke_herbceptions_return_failure_result<Fp>;
static_assert(__is_same(typename R4::value_type, int), "fp returns int");
static_assert(__is_same(typename R4::error_type, std::myerr), "fp return_failure{myerr}");

// __is_herbceptions_throws_constructible
struct throws_copy {
  throws_copy(throws_copy const &) throws;
};
struct noexcept_copy {
  noexcept_copy(noexcept_copy const &) noexcept;
};
struct throws_move {
  throws_move(throws_move &&) throws;
};
struct noexcept_move {
  noexcept_move(noexcept_move &&) noexcept;
};

static_assert(__is_herbceptions_throws_constructible(throws_copy, throws_copy const &),
              "throws_copy is throws constructible from const&");
static_assert(!__is_herbceptions_throws_constructible(noexcept_copy, noexcept_copy const &),
              "noexcept_copy is not throws constructible");
static_assert(__is_herbceptions_throws_constructible(throws_move, throws_move &&),
              "throws_move is throws constructible from &&");
static_assert(!__is_herbceptions_throws_constructible(noexcept_move, noexcept_move &&),
              "noexcept_move is not throws constructible");

// __is_herbceptions_throws_copy_constructible (via __is_herbceptions_throws_constructible)
static_assert(__is_herbceptions_throws_constructible(throws_copy, const throws_copy &),
              "throws_copy is throws copy constructible");
static_assert(!__is_herbceptions_throws_constructible(noexcept_copy, const noexcept_copy &),
              "noexcept_copy is not throws copy constructible");
static_assert(__is_herbceptions_throws_constructible(throws_move, throws_move &&),
              "throws_move is throws move constructible");
static_assert(!__is_herbceptions_throws_constructible(noexcept_move, noexcept_move &&),
              "noexcept_move is not throws move constructible");

// __is_herbceptions_throws_invocable
auto throws_func = []() throws {};
auto noexcept_func = []() noexcept {};
auto throws_int_func = []() throws -> int { return 0; };

static_assert(__is_herbceptions_throws_invocable(decltype(throws_func)),
              "throws lambda is throws invocable");
static_assert(!__is_herbceptions_throws_invocable(decltype(noexcept_func)),
              "noexcept lambda is not throws invocable");
static_assert(!__is_herbceptions_throws_invocable(int),
              "int is not invocable");

// __is_herbceptions_throws_invocable_r
static_assert(__is_herbceptions_throws_invocable_r(int, decltype(throws_int_func)),
              "throws_int_func is throws invocable_r int");
static_assert(__is_herbceptions_throws_invocable_r(void, decltype(throws_int_func)),
              "throws_int_func is throws invocable_r void");
static_assert(!__is_herbceptions_throws_invocable_r(int, decltype(throws_func)),
              "throws_func is not throws invocable_r int (void -> int)");
static_assert(!__is_herbceptions_throws_invocable_r(int, int),
              "int is not invocable_r int");

// Function pointers
using ThrowsFp = void (*)() throws;
using NoexceptFp = void (*)() noexcept;
using ThrowsIntFp = int (*)() throws;

static_assert(__is_herbceptions_throws_invocable(ThrowsFp),
              "throws fnptr is throws invocable");
static_assert(!__is_herbceptions_throws_invocable(NoexceptFp),
              "noexcept fnptr is not throws invocable");
static_assert(__is_herbceptions_throws_invocable_r(int, ThrowsIntFp),
              "throws int fnptr is throws invocable_r int");
static_assert(__is_herbceptions_throws_invocable_r(void, ThrowsIntFp),
              "throws int fnptr is throws invocable_r void");
static_assert(!__is_herbceptions_throws_invocable_r(int, ThrowsFp),
               "throws void fnptr is not throws invocable_r int");

// Function pointers with arguments
using ThrowsArgFp = void (*)(int, char *, char *) throws;
using NoexceptArgFp = void (*)(int, char *, char *) noexcept;
using PlainArgFp = void (*)(int, char *, char *);

static_assert(__is_herbceptions_throws_invocable(ThrowsArgFp, int, char *, char *),
              "throws fnptr with args is throws invocable");
static_assert(!__is_herbceptions_throws_invocable(NoexceptArgFp, int, char *, char *),
              "noexcept fnptr with args is not throws invocable");
static_assert(!__is_herbceptions_throws_invocable(PlainArgFp, int, char *, char *),
              "plain fnptr with args is not throws invocable");

// Pointer to function with arguments
using ThrowsArgFn = void(int, char *, char *) throws;
using NoexceptArgFn = void(int, char *, char *) noexcept;

static_assert(__is_herbceptions_throws_invocable(ThrowsArgFn *, int, char *, char *),
              "pointer to throws fn is throws invocable");
static_assert(!__is_herbceptions_throws_invocable(NoexceptArgFn *, int, char *, char *),
              "pointer to noexcept fn is not throws invocable");

// Herbceptions throws invocable with pointer return type
using ThrowsDataFp = void *(*)() throws;
using NoexceptDataFp = void *(*)() noexcept;

static_assert(__is_herbceptions_throws_invocable(ThrowsDataFp),
              "throws fnptr returning void* is throws invocable");
static_assert(!__is_herbceptions_throws_invocable(NoexceptDataFp),
              "noexcept fnptr returning void* is not throws invocable");

// Herbceptions throws invocable_r with pointer return type
static_assert(__is_herbceptions_throws_invocable_r(void *, ThrowsDataFp),
              "throws fnptr returning void* is throws invocable_r void*");
static_assert(!__is_herbceptions_throws_invocable_r(void *, NoexceptDataFp),
              "noexcept fnptr returning void* is not throws invocable_r void*");

// Herbceptions throws constructible with pointer return type
struct foo_throws {
  foo_throws(foo_throws &&) throws;
};
struct foo_noexcept {
  foo_noexcept(foo_noexcept &&) noexcept;
};

static_assert(__is_herbceptions_throws_constructible(foo_throws, foo_throws &&),
              "foo_throws is throws constructible from rvalue");
static_assert(!__is_herbceptions_throws_constructible(foo_noexcept, foo_noexcept &&),
              "foo_noexcept is not throws constructible from rvalue");

// Decay function pointer detection with custom stream types
namespace foo {

struct stream_observer {};
struct null_sink {};

// Decay function pointer types
using decay_fn = void *(stream_observer, char *, char *) throws;
using decay_noexcept_fn = void *(stream_observer, char *, char *) noexcept;

// Test direct usage (non-template)
static_assert(__is_herbceptions_throws_invocable(decay_fn, stream_observer, char *, char *),
              "decay_fn with stream_observer can throw");
static_assert(!__is_herbceptions_throws_invocable(decay_noexcept_fn, stream_observer, char *, char *),
              "decay_noexcept_fn cannot throw");

// Test with noexcept null_sink
using null_read_fn = void *(null_sink, char *, char *) noexcept;
static_assert(!__is_herbceptions_throws_invocable(null_read_fn, null_sink, char *, char *),
              "null_sink read cannot throw");

} // namespace foo

// Template variable with herbceptions trait (conditional throws pattern)
// Note: the trait checks if the function pointer type is declared throws AND
// if the arguments are convertible. Since decay_fn is throws, any
// convertible argument type will return true.
namespace bar {

using decay_fn = void *(int, char *, char *) throws;

template <typename Stream>
inline constexpr bool has_any_of_read_operations_herbceptions_throws =
    __is_herbceptions_throws_invocable(decay_fn, Stream, char *, char *);

static_assert(has_any_of_read_operations_herbceptions_throws<int>,
              "int read can throw");
static_assert(has_any_of_read_operations_herbceptions_throws<char>,
              "char read can throw (char converts to int)");

} // namespace bar
