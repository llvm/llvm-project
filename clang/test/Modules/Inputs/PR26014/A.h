#ifndef _LIBCPP_TYPE_TRAITS
#define _LIBCPP_TYPE_TRAITS


template <class _Tp>
struct underlying_type
{
    typedef __underlying_type(_Tp) type;
};

#endif  // _LIBCPP_TYPE_TRAITS

#include "B.h"
