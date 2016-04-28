#ifndef _LIBCPP_ALGORITHM
#define _LIBCPP_ALGORITHM
template <class _Tp, _Tp>
struct integral_constant {
  static const _Tp value = _Tp();
};

template <class _Tp>
struct is_nothrow_default_constructible
	: integral_constant<bool, __is_constructible(_Tp)> {};

template <class _Tp>
struct is_nothrow_move_constructible
    : integral_constant<bool, __is_constructible(_Tp, _Tp)> {};

class allocator {};
#endif
