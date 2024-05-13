// RUN: %clang_cc1 -x c++ -std=c++11 -triple x86_64-pc-linux -fsyntax-only
// RUN: %clang_cc1 -x c++ -std=c++11 -triple x86_64-windows-msvc -fsyntax-only

// Check we return non-zero values for supported attributes as per
// wg21.link/p2552r3.pdf
static_assert(__has_cpp_attribute(assume));

// The standard does not prescribe a behavior for [[carries_dependency]]

static_assert(__has_cpp_attribute(deprecated));
static_assert(__has_cpp_attribute(fallthrough));
static_assert(__has_cpp_attribute(likely));
static_assert(__has_cpp_attribute(unlikely));
static_assert(__has_cpp_attribute(maybe_unused));
static_assert(__has_cpp_attribute(nodiscard));
static_assert(__has_cpp_attribute(noreturn));

#ifdef _MSC_VER
// We do not support [[no_unique_address]] in MSVC emulation mode
static_assert(__has_cpp_attribute(no_unique_address) == 0);
#else
static_assert(__has_cpp_attribute(no_unique_address));
#endif
