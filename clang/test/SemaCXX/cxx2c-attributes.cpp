// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple x86_64-pc-linux -fsyntax-only -verify -Wno-c++17-extensions
// RUN: %clang_cc1 %s -x c++ -std=c++11 -triple x86_64-windows-msvc -fsyntax-only -verify=msvc -Wno-c++17-extensions
// expected-no-diagnostics

// Check we return non-zero values for supported attributes as per
// wg21.link/P2552
static_assert(__has_cpp_attribute(assume));

// [[carries_dependency]] was removed from the standard by P3475R2
static_assert(!__has_cpp_attribute(carries_dependency));

static_assert(__has_cpp_attribute(deprecated));
static_assert(__has_cpp_attribute(fallthrough));
static_assert(__has_cpp_attribute(likely));
static_assert(__has_cpp_attribute(unlikely));
static_assert(__has_cpp_attribute(maybe_unused));
static_assert(__has_cpp_attribute(nodiscard));
static_assert(__has_cpp_attribute(noreturn));

// We do not support [[no_unique_address]] in MSVC emulation mode
static_assert(__has_cpp_attribute(no_unique_address)); // msvc-error {{static assertion failed}}
