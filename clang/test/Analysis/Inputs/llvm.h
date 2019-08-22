#pragma clang system_header

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
} // namespace llvm
