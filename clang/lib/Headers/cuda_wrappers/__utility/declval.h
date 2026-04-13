#ifndef __CUDA_WRAPPERS_UTILITY_DECLVAL_H__
#define __CUDA_WRAPPERS_UTILITY_DECLVAL_H__

#include_next <__utility/declval.h>

// The stuff below is the exact copy of the <__utility/declval.h>,
// but with __device__ attribute applied to the functions, so it works on a GPU.

_LIBCPP_BEGIN_NAMESPACE_STD

// Suppress deprecation notice for volatile-qualified return type resulting
// from volatile-qualified types _Tp.
_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _Tp> __attribute__((device)) _Tp &&__declval(int);
template <class _Tp> __attribute__((device)) _Tp __declval(long);
_LIBCPP_SUPPRESS_DEPRECATED_POP

template <class _Tp>
__attribute__((device)) _LIBCPP_HIDE_FROM_ABI decltype(std::__declval<_Tp>(0))
declval() _NOEXCEPT {
  static_assert(!__is_same(_Tp, _Tp),
                "std::declval can only be used in an unevaluated context. "
                "It's likely that your current usage is trying to extract a "
                "value from the function.");
}

_LIBCPP_END_NAMESPACE_STD
#endif // __CUDA_WRAPPERS_UTILITY_DECLVAL_H__
