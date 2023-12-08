#include "a.h"
#ifndef _LIBCPP_VECTOR
template <class, class _Allocator>
class __vector_base {
protected:
  _Allocator __alloc() const;
  __vector_base(_Allocator);
};

template <class _Tp, class _Allocator = allocator>
class vector : __vector_base<_Tp, _Allocator> {
public:
  vector() noexcept(is_nothrow_default_constructible<_Allocator>::value);
  vector(const vector &);
  vector(vector &&)
      noexcept(is_nothrow_move_constructible<_Allocator>::value);
};

#endif
void GetUniquePtrType() { vector<char> v; }

