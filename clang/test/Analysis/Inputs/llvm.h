#pragma clang system_header

#include "system-header-simulator-cxx.h"

namespace llvm {
template <class X, class Y>
const X *cast(Y Value);

template <class X, class Y>
const X *dyn_cast(Y *Value);
template <class X, class Y>
const X &dyn_cast(Y &Value);

template <class X, class Y>
const X *cast_or_null(Y Value);

template <class X, class Y>
const X *dyn_cast_or_null(Y *Value);
template <class X, class Y>
const X *dyn_cast_or_null(Y &Value);

template <class X, class Y> inline bool isa(const Y &Val);

template <typename First, typename Second, typename... Rest, typename Y>
inline bool isa(const Y &Val) {
  return isa<First>(Val) || isa<Second, Rest...>(Val);
}

template <typename... X, class Y>
inline bool isa_and_nonnull(const Y &Val) {
  if (!Val)
    return false;
  return isa<X...>(Val);
}

template <typename X, typename Y>
std::unique_ptr<X> cast(std::unique_ptr<Y> &&Value);
} // namespace llvm
