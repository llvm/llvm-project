// libstdc++ uses the non-constexpr function std::__glibcxx_assert_fail()
// to trigger compilation errors when the __glibcxx_assert(cond) macro
// is used in a constexpr context.
// Compilation fails when using code from the libstdc++ (such as std::array) on
// device code, since these assertions invoke a non-constexpr host function from
// device code.
//
// To work around this issue, we declare our own device version of the function

#ifndef __CLANG_CUDA_WRAPPERS_BITS_CPP_CONFIG
#define __CLANG_CUDA_WRAPPERS_BITS_CPP_CONFIG

#include_next <bits/c++config.h>

#ifdef _LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_NAMESPACE_STD
#else
namespace std {
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_BEGIN_NAMESPACE_VERSION
#endif

#pragma push_macro("CUDA_NOEXCEPT")
#if __cplusplus >= 201103L
#define CUDA_NOEXCEPT noexcept
#else
#define CUDA_NOEXCEPT
#endif

__attribute__((device, noreturn)) inline void
__glibcxx_assert_fail(const char *file, int line, const char *function,
                      const char *condition) CUDA_NOEXCEPT {
#ifdef _GLIBCXX_VERBOSE_ASSERT
  if (file && function && condition)
    __builtin_printf("%s:%d: %s: Assertion '%s' failed.\n", file, line,
                     function, condition);
  else if (function)
    __builtin_printf("%s: Undefined behavior detected.\n", function);
#endif
  __builtin_abort();
}

#endif
__attribute__((device, noreturn, __always_inline__,
               __visibility__("default"))) inline void
__glibcxx_assert_fail() CUDA_NOEXCEPT {
  __builtin_abort();
}

#pragma pop_macro("CUDA_NOEXCEPT")

#ifdef _LIBCPP_END_NAMESPACE_STD
_LIBCPP_END_NAMESPACE_STD
#else
#ifdef _GLIBCXX_BEGIN_NAMESPACE_VERSION
_GLIBCXX_END_NAMESPACE_VERSION
#endif
} // namespace std
#endif

#endif
