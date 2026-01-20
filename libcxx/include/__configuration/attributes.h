//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CONFIGURATION_ATTRIBUTES_H
#define _LIBCPP___CONFIGURATION_ATTRIBUTES_H

#include <__config_site>
#include <__configuration/hardening.h>
#include <__configuration/language.h>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

#ifndef __has_declspec_attribute
#  define __has_declspec_attribute(__x) 0
#endif

// Attributes relevant for layout ABI
// ----------------------------------

#if __has_cpp_attribute(msvc::no_unique_address)
// MSVC implements [[no_unique_address]] as a silent no-op currently.
// (If/when MSVC breaks its C++ ABI, it will be changed to work as intended.)
// However, MSVC implements [[msvc::no_unique_address]] which does what
// [[no_unique_address]] is supposed to do, in general.
#  define _LIBCPP_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#  define _LIBCPP_NO_UNIQUE_ADDRESS [[__no_unique_address__]]
#endif

#define _LIBCPP_PACKED __attribute__((__packed__))

// Attributes affecting overload resolution
// ----------------------------------------

#if __has_attribute(__enable_if__)
#  define _LIBCPP_PREFERRED_OVERLOAD __attribute__((__enable_if__(true, "")))
#endif

// Visibility attributes
// ---------------------

#if defined(_LIBCPP_OBJECT_FORMAT_COFF)

#  ifdef _DLL
#    define _LIBCPP_CRT_FUNC __declspec(dllimport)
#  else
#    define _LIBCPP_CRT_FUNC
#  endif

#  if defined(_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS) || (defined(__MINGW32__) && !defined(_LIBCPP_BUILDING_LIBRARY))
#    define _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS
#    define _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS
#    define _LIBCPP_OVERRIDABLE_FUNC_VIS
#    define _LIBCPP_EXPORTED_FROM_ABI
#  elif defined(_LIBCPP_BUILDING_LIBRARY)
#    if defined(__MINGW32__)
#      define _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS __declspec(dllexport)
#      define _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS
#    else
#      define _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS
#      define _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS __declspec(dllexport)
#    endif
#    define _LIBCPP_OVERRIDABLE_FUNC_VIS __declspec(dllexport)
#    define _LIBCPP_EXPORTED_FROM_ABI __declspec(dllexport)
#  else
#    define _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS __declspec(dllimport)
#    define _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS
#    define _LIBCPP_OVERRIDABLE_FUNC_VIS
#    define _LIBCPP_EXPORTED_FROM_ABI __declspec(dllimport)
#  endif

#  define _LIBCPP_HIDDEN
#  define _LIBCPP_TEMPLATE_DATA_VIS
#  define _LIBCPP_NAMESPACE_VISIBILITY

#else

#  if !defined(_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
#    define _LIBCPP_VISIBILITY(vis) __attribute__((__visibility__(vis)))
#  else
#    define _LIBCPP_VISIBILITY(vis)
#  endif

#  define _LIBCPP_HIDDEN _LIBCPP_VISIBILITY("hidden")
#  define _LIBCPP_TEMPLATE_DATA_VIS _LIBCPP_VISIBILITY("default")
#  define _LIBCPP_EXPORTED_FROM_ABI _LIBCPP_VISIBILITY("default")
#  define _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS _LIBCPP_VISIBILITY("default")
#  define _LIBCPP_CLASS_TEMPLATE_INSTANTIATION_VIS

// TODO: Make this a proper customization point or remove the option to override it.
#  ifndef _LIBCPP_OVERRIDABLE_FUNC_VIS
#    define _LIBCPP_OVERRIDABLE_FUNC_VIS _LIBCPP_VISIBILITY("default")
#  endif

#  if !defined(_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS) && __has_attribute(__type_visibility__)
#    define _LIBCPP_NAMESPACE_VISIBILITY __attribute__((__type_visibility__("default")))
#  elif !defined(_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
#    define _LIBCPP_NAMESPACE_VISIBILITY __attribute__((__visibility__("default")))
#  else
#    define _LIBCPP_NAMESPACE_VISIBILITY
#  endif

#endif // defined(_LIBCPP_OBJECT_FORMAT_COFF)

// hide_from_abi
// -------------

#define _LIBCPP_ALWAYS_INLINE __attribute__((__always_inline__))

#if __has_attribute(exclude_from_explicit_instantiation)
#  define _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((__exclude_from_explicit_instantiation__))
#else
// Try to approximate the effect of exclude_from_explicit_instantiation
// (which is that entities are not assumed to be provided by explicit
// template instantiations in the dylib) by always inlining those entities.
#  define _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION _LIBCPP_ALWAYS_INLINE
#endif

#if _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_FAST
#  define _LIBCPP_HARDENING_SIG f
#elif _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_EXTENSIVE
#  define _LIBCPP_HARDENING_SIG s
#elif _LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG
#  define _LIBCPP_HARDENING_SIG d
#else
#  define _LIBCPP_HARDENING_SIG n // "none"
#endif

#if _LIBCPP_ASSERTION_SEMANTIC == _LIBCPP_ASSERTION_SEMANTIC_OBSERVE
#  define _LIBCPP_ASSERTION_SEMANTIC_SIG o
#elif _LIBCPP_ASSERTION_SEMANTIC == _LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE
#  define _LIBCPP_ASSERTION_SEMANTIC_SIG q
#elif _LIBCPP_ASSERTION_SEMANTIC == _LIBCPP_ASSERTION_SEMANTIC_ENFORCE
#  define _LIBCPP_ASSERTION_SEMANTIC_SIG e
#else
#  define _LIBCPP_ASSERTION_SEMANTIC_SIG i // `ignore`
#endif

#if !_LIBCPP_HAS_EXCEPTIONS
#  define _LIBCPP_EXCEPTIONS_SIG n
#else
#  define _LIBCPP_EXCEPTIONS_SIG e
#endif

#define _LIBCPP_ODR_SIGNATURE                                                                                          \
  _LIBCPP_CONCAT(                                                                                                      \
      _LIBCPP_CONCAT(_LIBCPP_CONCAT(_LIBCPP_HARDENING_SIG, _LIBCPP_ASSERTION_SEMANTIC_SIG), _LIBCPP_EXCEPTIONS_SIG),   \
      _LIBCPP_VERSION)

// This macro marks a symbol as being hidden from libc++'s ABI. This is achieved
// on two levels:
// 1. The symbol is given hidden visibility, which ensures that users won't start exporting
//    symbols from their dynamic library by means of using the libc++ headers. This ensures
//    that those symbols stay private to the dynamic library in which it is defined.
//
// 2. The symbol is given an ABI tag that encodes the ODR-relevant properties of the library.
//    This ensures that no ODR violation can arise from mixing two TUs compiled with different
//    versions or configurations of libc++ (such as exceptions vs no-exceptions). Indeed, if the
//    program contains two definitions of a function, the ODR requires them to be token-by-token
//    equivalent, and the linker is allowed to pick either definition and discard the other one.
//
//    For example, if a program contains a copy of `vector::at()` compiled with exceptions enabled
//    *and* a copy of `vector::at()` compiled with exceptions disabled (by means of having two TUs
//    compiled with different settings), the two definitions are both visible by the linker and they
//    have the same name, but they have a meaningfully different implementation (one throws an exception
//    and the other aborts the program). This violates the ODR and makes the program ill-formed, and in
//    practice what will happen is that the linker will pick one of the definitions at random and will
//    discard the other one. This can quite clearly lead to incorrect program behavior.
//
//    A similar reasoning holds for many other properties that are ODR-affecting. Essentially any
//    property that causes the code of a function to differ from the code in another configuration
//    can be considered ODR-affecting. In practice, we don't encode all such properties in the ABI
//    tag, but we encode the ones that we think are most important: library version, exceptions, and
//    hardening mode.
//
//    Note that historically, solving this problem has been achieved in various ways, including
//    force-inlining all functions or giving internal linkage to all functions. Both these previous
//    solutions suffer from drawbacks that lead notably to code bloat.
//
// Note that we use _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION to ensure that we don't depend
// on _LIBCPP_HIDE_FROM_ABI methods of classes explicitly instantiated in the dynamic library.
//
// Also note that the _LIBCPP_HIDE_FROM_ABI_VIRTUAL macro should be used on virtual functions
// instead of _LIBCPP_HIDE_FROM_ABI. That macro does not use an ABI tag. Indeed, the mangled
// name of a virtual function is part of its ABI, since some architectures like arm64e can sign
// the virtual function pointer in the vtable based on the mangled name of the function. Since
// we use an ABI tag that changes with each released version, the mangled name of the virtual
// function would change, which is incorrect. Note that it doesn't make much sense to change
// the implementation of a virtual function in an ABI-incompatible way in the first place,
// since that would be an ABI break anyway. Hence, the lack of ABI tag should not be noticeable.
//
// The macro can be applied to record and enum types. When the tagged type is nested in
// a record this "parent" record needs to have the macro too. Another use case for applying
// this macro to records and unions is to apply an ABI tag to inline constexpr variables.
// This can be useful for inline variables that are implementation details which are expected
// to change in the future.
//
// TODO: We provide a escape hatch with _LIBCPP_NO_ABI_TAG for folks who want to avoid increasing
//       the length of symbols with an ABI tag. In practice, we should remove the escape hatch and
//       use compression mangling instead, see https://github.com/itanium-cxx-abi/cxx-abi/issues/70.
#ifndef _LIBCPP_NO_ABI_TAG
#  define _LIBCPP_HIDE_FROM_ABI                                                                                        \
    _LIBCPP_HIDDEN _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION                                                         \
    __attribute__((__abi_tag__(_LIBCPP_TOSTRING(_LIBCPP_ODR_SIGNATURE))))
#else
#  define _LIBCPP_HIDE_FROM_ABI _LIBCPP_HIDDEN _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif
#define _LIBCPP_HIDE_FROM_ABI_VIRTUAL _LIBCPP_HIDDEN _LIBCPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION

// Optional attributes
// -------------------

// these are useful for a better QoI, but not required to be available

#define _LIBCPP_NOALIAS __attribute__((__malloc__))
#define _LIBCPP_NODEBUG [[__gnu__::__nodebug__]]
#define _LIBCPP_NO_SANITIZE(...) __attribute__((__no_sanitize__(__VA_ARGS__)))
#define _LIBCPP_INIT_PRIORITY_MAX __attribute__((__init_priority__(100)))
#define _LIBCPP_ATTRIBUTE_FORMAT(archetype, format_string_index, first_format_arg_index)                               \
  __attribute__((__format__(archetype, format_string_index, first_format_arg_index)))

#if __has_attribute(__no_sanitize__) && !defined(_LIBCPP_COMPILER_GCC)
#  define _LIBCPP_NO_CFI __attribute__((__no_sanitize__("cfi")))
#else
#  define _LIBCPP_NO_CFI
#endif

#if __has_attribute(__using_if_exists__)
#  define _LIBCPP_USING_IF_EXISTS __attribute__((__using_if_exists__))
#else
#  define _LIBCPP_USING_IF_EXISTS
#endif

#if __has_cpp_attribute(_Clang::__no_destroy__)
#  define _LIBCPP_NO_DESTROY [[_Clang::__no_destroy__]]
#else
#  define _LIBCPP_NO_DESTROY
#endif

#if __has_attribute(__diagnose_if__)
#  define _LIBCPP_DIAGNOSE_WARNING(...) __attribute__((__diagnose_if__(__VA_ARGS__, "warning")))
#else
#  define _LIBCPP_DIAGNOSE_WARNING(...)
#endif

#if __has_attribute(__diagnose_if__) && !defined(_LIBCPP_APPLE_CLANG_VER) &&                                           \
    (!defined(_LIBCPP_CLANG_VER) || _LIBCPP_CLANG_VER >= 2001)
#  define _LIBCPP_DIAGNOSE_IF(...) __attribute__((__diagnose_if__(__VA_ARGS__)))
#else
#  define _LIBCPP_DIAGNOSE_IF(...)
#endif

#define _LIBCPP_DIAGNOSE_NULLPTR_IF(condition, condition_description)                                                  \
  _LIBCPP_DIAGNOSE_IF(                                                                                                 \
      condition,                                                                                                       \
      "null passed to callee that requires a non-null argument" condition_description,                                 \
      "warning",                                                                                                       \
      "nonnull")

#if __has_cpp_attribute(_Clang::__lifetimebound__)
#  define _LIBCPP_LIFETIMEBOUND [[_Clang::__lifetimebound__]]
#else
#  define _LIBCPP_LIFETIMEBOUND
#endif

// This is to work around https://llvm.org/PR156809
#ifndef _LIBCPP_CXX03_LANG
#  define _LIBCPP_CTOR_LIFETIMEBOUND _LIBCPP_LIFETIMEBOUND
#else
#  define _LIBCPP_CTOR_LIFETIMEBOUND
#endif

#if __has_cpp_attribute(_Clang::__noescape__)
#  define _LIBCPP_NOESCAPE [[_Clang::__noescape__]]
#else
#  define _LIBCPP_NOESCAPE
#endif

#if __has_cpp_attribute(_Clang::__no_specializations__)
#  define _LIBCPP_NO_SPECIALIZATIONS                                                                                   \
    [[_Clang::__no_specializations__("Users are not allowed to specialize this standard library entity")]]
#else
#  define _LIBCPP_NO_SPECIALIZATIONS
#endif

#if __has_cpp_attribute(_Clang::__preferred_name__)
#  define _LIBCPP_PREFERRED_NAME(x) [[_Clang::__preferred_name__(x)]]
#else
#  define _LIBCPP_PREFERRED_NAME(x)
#endif

#if __has_cpp_attribute(_Clang::__scoped_lockable__)
#  define _LIBCPP_SCOPED_LOCKABLE [[_Clang::__scoped_lockable__]]
#else
#  define _LIBCPP_SCOPED_LOCKABLE
#endif

#if __has_cpp_attribute(_Clang::__capability__)
#  define _LIBCPP_CAPABILITY(...) [[_Clang::__capability__(__VA_ARGS__)]]
#else
#  define _LIBCPP_CAPABILITY(...)
#endif

#if __has_attribute(__acquire_capability__)
#  define _LIBCPP_ACQUIRE_CAPABILITY(...) __attribute__((__acquire_capability__(__VA_ARGS__)))
#else
#  define _LIBCPP_ACQUIRE_CAPABILITY(...)
#endif

#if __has_cpp_attribute(_Clang::__try_acquire_capability__)
#  define _LIBCPP_TRY_ACQUIRE_CAPABILITY(...) [[_Clang::__try_acquire_capability__(__VA_ARGS__)]]
#else
#  define _LIBCPP_TRY_ACQUIRE_CAPABILITY(...)
#endif

#if __has_cpp_attribute(_Clang::__acquire_shared_capability__)
#  define _LIBCPP_ACQUIRE_SHARED_CAPABILITY [[_Clang::__acquire_shared_capability__]]
#else
#  define _LIBCPP_ACQUIRE_SHARED_CAPABILITY
#endif

#if __has_cpp_attribute(_Clang::__try_acquire_shared_capability__)
#  define _LIBCPP_TRY_ACQUIRE_SHARED_CAPABILITY(...) [[_Clang::__try_acquire_shared_capability__(__VA_ARGS__)]]
#else
#  define _LIBCPP_TRY_ACQUIRE_SHARED_CAPABILITY(...)
#endif

#if __has_cpp_attribute(_Clang::__release_capability__)
#  define _LIBCPP_RELEASE_CAPABILITY [[_Clang::__release_capability__]]
#else
#  define _LIBCPP_RELEASE_CAPABILITY
#endif

#if __has_cpp_attribute(_Clang::__release_shared_capability__)
#  define _LIBCPP_RELEASE_SHARED_CAPABILITY [[_Clang::__release_shared_capability__]]
#else
#  define _LIBCPP_RELEASE_SHARED_CAPABILITY
#endif

#if __has_attribute(__requires_capability__)
#  define _LIBCPP_REQUIRES_CAPABILITY(...) __attribute__((__requires_capability__(__VA_ARGS__)))
#else
#  define _LIBCPP_REQUIRES_CAPABILITY(...)
#endif

#if __has_cpp_attribute(_Clang::__no_thread_safety_analysis__)
#  define _LIBCPP_NO_THREAD_SAFETY_ANALYSIS [[_Clang::__no_thread_safety_analysis__]]
#else
#  define _LIBCPP_NO_THREAD_SAFETY_ANALYSIS
#endif

#if defined(_LIBCPP_ABI_MICROSOFT) && __has_declspec_attribute(empty_bases)
#  define _LIBCPP_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#else
#  define _LIBCPP_DECLSPEC_EMPTY_BASES
#endif

// Allow for build-time disabling of unsigned integer sanitization
#if __has_attribute(no_sanitize) && !defined(_LIBCPP_COMPILER_GCC)
#  define _LIBCPP_DISABLE_UBSAN_UNSIGNED_INTEGER_CHECK __attribute__((__no_sanitize__("unsigned-integer-overflow")))
#else
#  define _LIBCPP_DISABLE_UBSAN_UNSIGNED_INTEGER_CHECK
#endif

#if __has_feature(nullability)
#  define _LIBCPP_DIAGNOSE_NULLPTR _Nonnull
#else
#  define _LIBCPP_DIAGNOSE_NULLPTR
#endif

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
// The CUDA SDK contains an unfortunate definition for the __noinline__ macro,
// which breaks the regular __attribute__((__noinline__)) syntax. Therefore,
// when compiling for CUDA we use the non-underscored version of the noinline
// attribute.
//
// This is a temporary workaround and we still expect the CUDA SDK team to solve
// this issue properly in the SDK headers.
//
// See https://github.com/llvm/llvm-project/pull/73838 for more details.
#  define _LIBCPP_NOINLINE __attribute__((noinline))
#elif __has_attribute(__noinline__)
#  define _LIBCPP_NOINLINE __attribute__((__noinline__))
#else
#  define _LIBCPP_NOINLINE
#endif

// Deprecation macros
// ------------------

// Deprecations warnings are always enabled, except when users explicitly opt-out
// by defining _LIBCPP_DISABLE_DEPRECATION_WARNINGS.
#if !defined(_LIBCPP_DISABLE_DEPRECATION_WARNINGS)
#  if __has_attribute(__deprecated__)
#    define _LIBCPP_DEPRECATED __attribute__((__deprecated__))
#    define _LIBCPP_DEPRECATED_(m) __attribute__((__deprecated__(m)))
#  elif _LIBCPP_STD_VER >= 14
#    define _LIBCPP_DEPRECATED [[deprecated]]
#    define _LIBCPP_DEPRECATED_(m) [[deprecated(m)]]
#  else
#    define _LIBCPP_DEPRECATED
#    define _LIBCPP_DEPRECATED_(m)
#  endif
#else
#  define _LIBCPP_DEPRECATED
#  define _LIBCPP_DEPRECATED_(m)
#endif

#if !defined(_LIBCPP_CXX03_LANG)
#  define _LIBCPP_DEPRECATED_IN_CXX11 _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_IN_CXX11
#endif

#if _LIBCPP_STD_VER >= 14
#  define _LIBCPP_DEPRECATED_IN_CXX14 _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_IN_CXX14
#endif

#if _LIBCPP_STD_VER >= 17
#  define _LIBCPP_DEPRECATED_IN_CXX17 _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_IN_CXX17
#endif

#if _LIBCPP_STD_VER >= 20
#  define _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_IN_CXX20
#endif

#if _LIBCPP_STD_VER >= 23
#  define _LIBCPP_DEPRECATED_IN_CXX23 _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_IN_CXX23
#endif

#if _LIBCPP_STD_VER >= 26
#  define _LIBCPP_DEPRECATED_IN_CXX26 _LIBCPP_DEPRECATED
#  define _LIBCPP_DEPRECATED_IN_CXX26_(m) _LIBCPP_DEPRECATED_(m)
#else
#  define _LIBCPP_DEPRECATED_IN_CXX26
#  define _LIBCPP_DEPRECATED_IN_CXX26_(m)
#endif

#if _LIBCPP_HAS_CHAR8_T
#  define _LIBCPP_DEPRECATED_WITH_CHAR8_T _LIBCPP_DEPRECATED
#else
#  define _LIBCPP_DEPRECATED_WITH_CHAR8_T
#endif

#endif // _LIBCPP___CONFIGURATION_ATTRIBUTES_H
