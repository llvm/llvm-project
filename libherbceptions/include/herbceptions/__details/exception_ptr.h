//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <exception>

namespace std {

// The compiler (-fherbceptions) emits direct references to the
// __cxa_error_code_*/__cxa_error_domain_* symbols above when catching
// legacy C++ exceptions; it does not consult this specialization on that
// path. The specialization remains for generic machinery and for
// non-patched compilers.
template <> class error_domain<::std::exception_ptr> {
public:
  using errc_type = ::std::exception_ptr;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
#ifdef _MSC_VER
    return ::std::error_domains::__cxa_error_domain_msvc_exception_ptr();
#else
    return ::std::error_domains::__cxa_error_domain_itanium_exception_ptr();
#endif
  }
  static inline ::std::size_t code(errc_type const &__e) noexcept {
#ifdef _MSC_VER
    return ::std::error_domains::__cxa_error_code_msvc_exception_ptr_clone(
        __builtin_addressof(__e));
#else
    void *__temp;
    __builtin_memcpy(__builtin_addressof(__temp), __builtin_addressof(__e),
                     sizeof(void *));
    return ::std::error_domains::__cxa_error_code_itanium_exception_ptr_clone(
        __temp);
#endif
  }
};

} // namespace std
