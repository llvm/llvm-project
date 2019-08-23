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

template <class X, class Y>
bool isa(Y Value);

template <class X, class Y>
bool isa_and_nonnull(Y Value);

template <typename X, typename Y>
std::unique_ptr<X> cast(std::unique_ptr<Y> &&Value);
} // namespace llvm
