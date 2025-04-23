#ifndef _LIBCPP___CONCEPTS_CORE_CONVERTIBLE_TO_H
#define _LIBCPP___CONCEPTS_CORE_CONVERTIBLE_TO_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

// [conv.general]/3 says "E is convertible to T" whenever "T t=E;" is well-formed.
// We can't test for that, but we can test implicit convertibility by passing it
// to a function. Unlike std::convertible_to, __core_convertible_to doesn't test
// static_cast or handle cv void, while accepting non-movable types.
//
// This is a conceptual __is_core_convertible.
template <class _Tp, class _Up>
concept __core_convertible_to = requires {
  // rejects function and array types which are adjusted to pointer types in parameter lists
  static_cast<_Up (*)()>(nullptr)();
  static_cast<void (*)(_Up)>(nullptr)(static_cast<_Tp (*)()>(nullptr)());
};

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CONCEPTS_CORE_CONVERTIBLE_TO_H
