#ifndef THIRD_PARTY_LLVM_LLVM_PROJECT_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_DISJUNCTION_H_
#define THIRD_PARTY_LLVM_LLVM_PROJECT_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_DISJUNCTION_H_

#include "src/__support/CPP/type_traits/conditional.h"
#include "src/__support/CPP/type_traits/true_type.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

template <typename...>
struct disjunction : true_type {};
template <typename B>
struct disjunction<B> : B {};
template <typename B1, typename... Bs>
struct disjunction<B1, Bs...>
    : conditional_t<bool(B1::value), B1, disjunction<Bs...>> {};

template <typename... Bs>
constexpr bool disjunction_v = disjunction<Bs...>::value;

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif  // THIRD_PARTY_LLVM_LLVM_PROJECT_LIBC_SRC___SUPPORT_CPP_TYPE_TRAITS_DISJUNCTION_H_
