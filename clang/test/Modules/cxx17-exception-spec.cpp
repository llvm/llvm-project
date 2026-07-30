// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fno-modules-error-recovery -fno-spell-checking -verify -std=c++17 %s

#pragma clang module build a
module a {}
#pragma clang module contents
#pragma clang module begin a
constexpr bool return_true() { return true; }
struct X {
  static void f() noexcept(return_true());
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build b
module b {}
#pragma clang module contents
#pragma clang module begin b
#pragma clang module import a
using T = decltype(return_true() && noexcept(X::f()));
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import a
#pragma clang module import b

// Trigger import of return_true and then X::f in the same pass. This causes
// the type of X::f to be loaded while we have a pending body for return_true,
// which means evaluation of its exception specification at that point would
// fail.
T t;

static_assert(noexcept(X().f()));

// expected-no-diagnostics
