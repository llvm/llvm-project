// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
inline namespace {
template <int __v> struct integral_constant {
  static const int value = __v;
};
template <bool, class> using __enable_if_t = int;
char *addressof(char &);
struct pointer_traits {
  template <class _Up> using rebind = _Up *;
};
} // namespace
} // namespace std
void *operator new(decltype(sizeof(int)), void *);
namespace std {
inline namespace {
template <class _Tp> using __make_unsigned_t = __make_unsigned(_Tp);
template <class _Default, template <class> class>
using __detected_or_t = _Default;
template <class _Tp> using __pointer_member = _Tp;
template <class _Tp, class>
using __pointer = __detected_or_t<_Tp *, __pointer_member>;
template <class _Tp> using __size_type_member = _Tp;
template <class, class _DiffType>
using __size_type =
    __detected_or_t<__make_unsigned_t<_DiffType>, __size_type_member>;
struct allocation_result {
  char *ptr;
  unsigned long count;
};
template <class _Alloc> struct allocator_traits {
  using allocator_type = _Alloc;
  using pointer =
      __pointer<typename allocator_type::value_type, allocator_type>;
  using const_pointer = pointer_traits::rebind<char>;
  using size_type =
      __size_type<allocator_type, decltype(static_cast<int *>(nullptr) -
                                           static_cast<int *>(nullptr))>;
  template <class _Ap>
  static constexpr allocation_result allocate_at_least(_Ap __alloc,
                                                       size_type __n) {
    return {__alloc.allocate(__n), (unsigned long)__n};
  }
};
template <class _Alloc>
constexpr auto __allocate_at_least(_Alloc __alloc, decltype(sizeof(int)) __n) {
  return allocator_traits<_Alloc>::allocate_at_least(__alloc, __n);
}
template <class> struct allocator {
  typedef char value_type;
  constexpr char *allocate(decltype(sizeof(int)) __n) {
    return static_cast<char *>(operator new(__n));
  }
  constexpr void deallocate(char *__p) { operator delete(__p); }
};
struct __long {
  allocator_traits<allocator<char>>::size_type __is_long_;
  allocator_traits<allocator<char>>::size_type __size_;
  allocator_traits<allocator<char>>::pointer __data_;
};
allocator<char> __alloc_;
struct basic_string {
  __long __l;
  constexpr basic_string(basic_string &__str) {
    allocator_traits<allocator<char>>::size_type __trans_tmp_1 =
        __str.__get_long_size();
    auto __allocation = __allocate_at_least(__alloc_, __trans_tmp_1);
    for (allocator_traits<allocator<char>>::size_type __i = 0;
         __i != __allocation.count; ++__i) {
      char *__trans_tmp_9 = addressof(__allocation.ptr[__i]);
      new (__trans_tmp_9) char();
    }
    __l.__data_ = __allocation.ptr;
    __l.__is_long_ = __l.__size_ = __trans_tmp_1;
  }
  template <__enable_if_t<integral_constant<false>::value, int> = 0>
  constexpr basic_string(const char *__s, allocator<char>) {
    decltype(sizeof(int)) __trans_tmp_11, __i = 0;
    for (; __s[__i]; ++__i)
      __trans_tmp_11 = __i;
    auto __allocation = __allocate_at_least(__alloc_, 1);
    __l.__data_ = __allocation.ptr;
    __l.__size_ = __trans_tmp_11;
  }
  constexpr ~basic_string() {
    allocator<char> __a;
    __a.deallocate(__l.__data_);
  }
  constexpr allocator_traits<allocator<char>>::size_type size() {
    return __l.__is_long_;
  }
  constexpr char *data() {
    allocator_traits<allocator<char>>::const_pointer __trans_tmp_6 =
        __l.__is_long_ ? __l.__data_ : 0;
    return __trans_tmp_6;
  }
  constexpr allocator_traits<allocator<char>>::size_type __get_long_size() {
    return __l.__size_;
  }
};
constexpr void operator==(basic_string __lhs, basic_string __rhs) {
  decltype(sizeof(int)) __lhs_sz = __lhs.size();
  char *__trans_tmp_10 = __rhs.data(), *__trans_tmp_15 = __lhs.data();
  __builtin_memcmp(__trans_tmp_15, __trans_tmp_10, __lhs_sz);
}
} // namespace
} // namespace std
constexpr void test(std::basic_string s0) {
  std::basic_string s1 = s0, s2(s0);
  s2 == s1;
}
constexpr bool test() {
  test(std::basic_string("2345678901234567890", std::allocator<char>()));
  return true;
}
static_assert(test());
