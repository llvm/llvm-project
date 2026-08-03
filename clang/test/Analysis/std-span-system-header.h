#pragma clang system_header

#include "Inputs/system-header-simulator-cxx.h"

namespace std {

template <class _Tp, size_t _Size>
struct array {
  typedef array                                 __self;
  typedef _Tp                                   value_type;
  typedef value_type&                           reference;
  typedef const value_type&                     const_reference;
  typedef value_type*                           iterator;
  typedef const value_type*                     const_iterator;
  typedef value_type*                           pointer;
  typedef const value_type*                     const_pointer;
  typedef size_t                                size_type;

  _Tp __elems_[_Size];

  reference operator[](size_type __n);
  const_reference operator[](size_type __n) const;

  value_type* data();
  const value_type* data() const;

  size_type size() const;

  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
};

inline constexpr size_t dynamic_extent = (size_t)-1;

template <typename _Tp, size_t _Extent = dynamic_extent>
class span {
public:
  using element_type           = _Tp;
  using value_type             = remove_cv_t<_Tp>;
  using size_type              = size_t;
  using pointer                = _Tp *;

  template <size_t _Sz = _Extent> requires(_Sz == 0)
  constexpr span();

  constexpr span(const span&);

  template <typename _It>
  constexpr span(_It* __first, size_type __count);

  template <size_t _Sz>
  constexpr span(element_type (&__arr)[_Sz]);

  template <class _OtherElementType>
  constexpr span(array<_OtherElementType, _Extent>& __arr);

  template <class _OtherElementType>
  constexpr span(const array<_OtherElementType, _Extent>& __arr);

  constexpr pointer data() const;

  constexpr size_type size() const;
  constexpr size_type size_bytes() const;

  constexpr span<element_type, dynamic_extent> first(size_type __count) const;
};

template<class _Tp, size_t _Sz>
  span(_Tp (&)[_Sz]) -> span<_Tp, _Sz>;

template<class _Tp, size_t _Sz>
  span(array<_Tp, _Sz>&) -> span<_Tp, _Sz>;

template<class _Tp, size_t _Sz>
  span(const array<_Tp, _Sz>&) -> span<const _Tp, _Sz>;
}
