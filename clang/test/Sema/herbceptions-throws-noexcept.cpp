// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fsyntax-only -verify %s

// 'throws' supersedes 'noexcept': the two cannot be combined in either order.
// Any 'throws' followed by 'noexcept' (or vice versa) is rejected.

void a() throws noexcept(false); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void b() noexcept(false) throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void c() throws(true) noexcept(false); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void d() noexcept(false) throws(true); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void e() throws noexcept(true); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void f() noexcept(true) throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void g() noexcept throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
void h() throws noexcept; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}

// throws(false) means the function cannot fail: it becomes noexcept.
void i() throws(false);

// Member functions (delayed exception spec parsing) behave identically.
struct S {
  void a() throws noexcept(false); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
  void b() noexcept(false) throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
  void c() noexcept(true) throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
  void d() throws noexcept(true); // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
  void e() noexcept throws; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
};

// Lambdas behave identically.
auto l1 = []() noexcept(true) throws {}; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
auto l2 = []() throws noexcept {}; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}
auto l3 = []() noexcept(false) throws {}; // expected-error {{'throws'/'return_failure' and 'noexcept' cannot be combined}}

// throws(expr) expression operator: detects whether an expression can propagate
// a herbception error through the implicit `throws` channel.
void throws_fn() throws;
void return_failure_fn() return_failure{int};
void plain_fn() noexcept;

static_assert(throws(throws_fn()), "throws function should be detected");
static_assert(!throws(return_failure_fn()), "return_failure{E} function should NOT be detected");
static_assert(!throws(plain_fn()), "noexcept function should NOT be detected");

// throws(expr) works in constant expressions.
constexpr bool test_throws() {
  return throws(throws_fn());
}
static_assert(test_throws(), "throws in constexpr");

// FFI boundary: throw throws cxx_std_error{domain, code} is allowed — the
// global struct cxx_std_error ({void*, uintptr_t}) is layout-compatible with
// std::error and passes through without going through error_domain.
struct cxx_std_error { void *domain; __SIZE_TYPE__ code; };

extern "C" struct cxx_std_error get_c_error();

void ffi_throws() throws {
  struct cxx_std_error e = get_c_error();
  throw throws e;
}
