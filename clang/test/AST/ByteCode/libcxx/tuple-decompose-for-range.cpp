// RUN: %clang_cc1 -std=c++2c -verify=expected,both %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++2c -verify=ref,both      %s

// both-no-diagnostics

namespace std {
typedef __SIZE_TYPE__ size_t;
}
extern "C++" {
namespace std {
template <typename> struct iterator_traits;
}
}
namespace std {
template <typename _Tp, _Tp __v> struct integral_constant {
  static constexpr _Tp value = __v;
};
template <bool __v> using __bool_constant = integral_constant<bool, __v>;
using false_type = __bool_constant<false>;
template <bool, typename _Tp = void> struct enable_if {
  using type = _Tp;
};
template <bool> struct __conditional {
  template <typename _Tp, typename> using type = _Tp;
};
template <bool _Cond, typename _If, typename _Else>
using __conditional_t =
    typename __conditional<_Cond>::template type<_If, _Else>;
template <typename _Tp>
struct is_empty : public __bool_constant<__is_empty(_Tp)> {};
template <typename _Tp, typename _Up>
struct is_same : public __bool_constant<__is_same(_Tp, _Up)> {};
template <typename _Tp> struct remove_cv {
  using type = __remove_cv(_Tp);
};
template <typename _Tp> struct tuple_size;
template <typename _Tp, typename _Up = typename remove_cv<_Tp>::type,
          typename = typename enable_if<is_same<_Tp, _Up>::value>::type,
          size_t = tuple_size<_Tp>::value>
using __enable_if_has_tuple_size = _Tp;
template <typename _Tp>
struct tuple_size<const __enable_if_has_tuple_size<_Tp>>
    : public tuple_size<_Tp> {};
template <size_t __i, typename _Tp> struct tuple_element;
template <size_t __i, typename _Tp>
using __tuple_element_t = typename tuple_element<__i, _Tp>::type;
template <size_t __i, typename _Tp> struct tuple_element<__i, const _Tp> {
  using type = const __tuple_element_t<__i, _Tp>;
};
template <size_t _Np, typename... _Types> struct _Nth_type {
  using type = __type_pack_element<_Np, _Types...>;
};
template <typename _Tp> struct iterator_traits<_Tp *> {
  using reference = _Tp &;
};
} // namespace std
extern "C++" {
void *operator new(std::size_t, void *__p);
}
namespace std {
template <typename _Tp, typename... _Args>
constexpr inline void _Construct(_Tp *__p, _Args &&...__args) {
  ::new (__p) _Tp(__args...);
};
} // namespace std
namespace __gnu_cxx {
template <typename _Iterator, typename _Container> class __normal_iterator {
protected:
  _Iterator _M_current;
  typedef std::iterator_traits<_Iterator> __traits_type;

public:
  typedef _Iterator iterator_type;
  typedef typename __traits_type::reference reference;
  explicit constexpr __normal_iterator(const _Iterator &__i)
      : _M_current(__i) {};
  constexpr reference operator*() const { return *_M_current; }
  constexpr __normal_iterator &operator++() {
    ++_M_current;
    return *this;
  }
  constexpr const _Iterator &base() const { return _M_current; }
};
template <typename _Iterator, typename _Container>
constexpr bool
operator==(const __normal_iterator<_Iterator, _Container> &__lhs,
           const __normal_iterator<_Iterator, _Container> &__rhs) {
  return __lhs.base() == __rhs.base();
}
} // namespace __gnu_cxx
namespace std {
template <typename _Tp> class __new_allocator {};
template <typename _Tp> using __allocator_base = __new_allocator<_Tp>;
template <typename> struct allocator_traits;
template <typename _Tp> class allocator : public __allocator_base<_Tp> {
public:
  typedef _Tp value_type;
  constexpr _Tp *allocate(size_t __n) {
    __n *= sizeof(_Tp);
    return static_cast<_Tp *>(::operator new(__n));
  }
  constexpr void deallocate(_Tp *__p, size_t __n) { ::operator delete(__p); }
};
template <typename _Tp> struct allocator_traits<allocator<_Tp>> {
  using allocator_type = allocator<_Tp>;
  using pointer = _Tp *;
  using size_type = std::size_t;
  template <typename _Up> using rebind_alloc = allocator<_Up>;
  static constexpr pointer allocate(allocator_type &__a, size_type __n) {
    return __a.allocate(__n);
  }
  static constexpr void deallocate(allocator_type &__a, pointer __p,
                                   size_type __n) {
    __a.deallocate(__p, __n);
  }
};
} // namespace std
namespace __gnu_cxx {
template <typename _Alloc, typename = typename _Alloc::value_type>
struct __alloc_traits : std::allocator_traits<_Alloc> {
  typedef std::allocator_traits<_Alloc> _Base_type;
  template <typename _Tp> struct rebind {
    typedef typename _Base_type::template rebind_alloc<_Tp> other;
  };
};
} // namespace __gnu_cxx
namespace std {
template <typename _InputIterator, typename _ForwardIterator>
constexpr _ForwardIterator __do_uninit_copy(_InputIterator __first,
                                            _InputIterator __last,
                                            _ForwardIterator __result) {
  _ForwardIterator __cur = __result;
  for (; __first != __last; ++__first, ++__cur)
    std::_Construct(&*__cur, *__first);
  return __cur;
};
template <typename _InputIterator, typename _ForwardIterator, typename _Tp>
constexpr inline _ForwardIterator
__uninitialized_copy_a(_InputIterator __first, _InputIterator __last,
                       _ForwardIterator __result, allocator<_Tp> &) {
  return std::__do_uninit_copy(__first, __last, __result);
}
template <typename _Tp, typename _Alloc> struct _Vector_base {
  typedef
      typename __gnu_cxx::__alloc_traits<_Alloc>::template rebind<_Tp>::other
          _Tp_alloc_type;
  typedef typename __gnu_cxx::__alloc_traits<_Tp_alloc_type>::pointer pointer;
  struct _Vector_impl_data {
    pointer _M_start;
    pointer _M_finish;
    pointer _M_end_of_storage;
  };
  struct _Vector_impl : public _Tp_alloc_type, public _Vector_impl_data {};

public:
  typedef _Alloc allocator_type;
  constexpr _Tp_alloc_type &_M_get_Tp_allocator() { return this->_M_impl; }
  constexpr _Vector_base(const allocator_type &__a) : _M_impl(__a) {}
  constexpr ~_Vector_base() {
    _M_deallocate(_M_impl._M_start,
                  _M_impl._M_end_of_storage - _M_impl._M_start);
  }

public:
  _Vector_impl _M_impl;
  constexpr pointer _M_allocate(size_t __n) {
    typedef __gnu_cxx::__alloc_traits<_Tp_alloc_type> _Tr;
    return __n != 0 ? _Tr::allocate(_M_impl, __n) : pointer();
  }
  constexpr void _M_deallocate(pointer __p, size_t __n) {
    typedef __gnu_cxx::__alloc_traits<_Tp_alloc_type> _Tr;
    if (__p)
      _Tr::deallocate(_M_impl, __p, __n);
  }

protected:
};
template <typename _Tp, typename _Alloc = std::allocator<_Tp>>
class vector : protected _Vector_base<_Tp, _Alloc> {
  typedef _Vector_base<_Tp, _Alloc> _Base;

public:
  typedef _Tp value_type;
  typedef typename _Base::pointer pointer;
  typedef __gnu_cxx::__normal_iterator<pointer, vector> iterator;
  typedef size_t size_type;
  typedef _Alloc allocator_type;
  using _Base::_M_get_Tp_allocator;

public:
private:
public:
  constexpr vector(_Tp __l, const allocator_type &__a = allocator_type()) : _Base(__a) {

  }
  constexpr iterator begin() { return iterator(this->_M_impl._M_start); }
  constexpr iterator end() { return iterator(this->_M_impl._M_finish); }

protected:
  template <typename _Iterator>
  constexpr void _M_range_initialize_n(_Iterator __first, _Iterator __last,
                                       size_type __n) {
    pointer __start = this->_M_impl._M_start = this->_M_allocate(((__n)));
    this->_M_impl._M_end_of_storage = __start + __n;
    this->_M_impl._M_finish = std::__uninitialized_copy_a(
        __first, __last, __start, _M_get_Tp_allocator());
  }
};
template <typename _Tp> struct __is_empty_non_tuple : is_empty<_Tp> {};
template <typename _Tp>
using __empty_not_final =
    __conditional_t<__is_final(_Tp), false_type, __is_empty_non_tuple<_Tp>>;
template <size_t _Idx, typename _Head, bool = __empty_not_final<_Head>::value>
struct _Head_base;
template <size_t _Idx, typename _Head> struct _Head_base<_Idx, _Head, false> {
  static constexpr const _Head &_M_head(const _Head_base &__b) {
    return __b._M_head_impl;
  }
  _Head _M_head_impl;
};
template <size_t _Idx, typename... _Elements> struct _Tuple_impl;
template <size_t _Idx, typename _Head>
struct _Tuple_impl<_Idx, _Head> : private _Head_base<_Idx, _Head> {
  typedef _Head_base<_Idx, _Head> _Base;
  static constexpr const _Head &_M_head(const _Tuple_impl &__t) {
    return _Base::_M_head(__t);
  }
  explicit constexpr _Tuple_impl(const _Head &__head) : _Base(__head) {}

protected:
};
template <typename... _Elements>
class tuple : public _Tuple_impl<0, _Elements...> {
  using _Inherited = _Tuple_impl<0, _Elements...>;

public:
  template <typename = void>
  constexpr tuple(const _Elements &...__elements) : _Inherited(__elements...) {}
};
template <typename... _Elements>
struct tuple_size<tuple<_Elements...>>
    : public integral_constant<size_t, sizeof...(_Elements)> {};
template <size_t __i, typename... _Types>
struct tuple_element<__i, tuple<_Types...>> {
  using type = typename _Nth_type<__i, _Types...>::type;
};
template <size_t __i, typename _Head, typename... _Tail>
constexpr const _Head &
__get_helper(const _Tuple_impl<__i, _Head, _Tail...> &__t) {
  return _Tuple_impl<__i, _Head, _Tail...>::_M_head(__t);
};
template <size_t __i, typename... _Elements>
constexpr const int get(const tuple<_Elements...> &&__t) {
  return std::__get_helper<__i>(__t);
};
} // namespace std
constexpr int foo() {
  std::vector<std::tuple<int>> data_tuples = {{1}};
  for (const auto [id] : data_tuples) {
    int a = id + 3;
  }
  return 1;
}
static_assert(foo() == 1);
