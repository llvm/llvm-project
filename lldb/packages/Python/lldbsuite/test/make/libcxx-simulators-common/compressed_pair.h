#ifndef STD_LLDB_COMPRESSED_PAIR_H
#define STD_LLDB_COMPRESSED_PAIR_H

#include <type_traits>
#include <utility> // for std::forward

// COMPRESSED_PAIR_REV versions:
// 0 -> Post-c88580c layout
// 1 -> Post-27c83382d83dc layout
// 2 -> Post-769c42f4a552a layout
// 3 -> padding-less no_unique_address-based layout (introduced in 27c83382d83dc)

namespace std {
namespace __lldb {

#if __has_cpp_attribute(msvc::no_unique_address)
#define _LLDB_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif __has_cpp_attribute(no_unique_address)
#define _LLDB_NO_UNIQUE_ADDRESS [[__no_unique_address__]]
#endif

// From libc++ datasizeof.h
template <class _Tp> struct _FirstPaddingByte {
  _LLDB_NO_UNIQUE_ADDRESS _Tp __v_;
  char __first_padding_byte_;
};

template <class _Tp>
inline const size_t __datasizeof_v =
    __builtin_offsetof(_FirstPaddingByte<_Tp>, __first_padding_byte_);

template <class _Tp>
struct __lldb_is_final : public integral_constant<bool, __is_final(_Tp)> {};

// The legacy layout has been patched, see
// https://github.com/llvm/llvm-project/pull/142516.
#if COMPRESSED_PAIR_REV == 1
template <class _ToPad> class __compressed_pair_padding {
  char __padding_[((is_empty<_ToPad>::value &&
                    !__lldb_is_final<_ToPad>::value) ||
                   is_reference<_ToPad>::value)
                      ? 0
                      : sizeof(_ToPad) - __datasizeof_v<_ToPad>];
};
#elif COMPRESSED_PAIR_REV > 1 && COMPRESSED_PAIR_REV < 3
template <class _ToPad>
inline const bool __is_reference_or_unpadded_object =
    (std::is_empty<_ToPad>::value && !__lldb_is_final<_ToPad>::value) ||
    sizeof(_ToPad) == __datasizeof_v<_ToPad>;

template <class _Tp>
inline const bool __is_reference_or_unpadded_object<_Tp &> = true;

template <class _Tp>
inline const bool __is_reference_or_unpadded_object<_Tp &&> = true;

template <class _ToPad, bool _Empty = __is_reference_or_unpadded_object<_ToPad>>
class __compressed_pair_padding {
  char __padding_[sizeof(_ToPad) - __datasizeof_v<_ToPad>] = {};
};

template <class _ToPad> class __compressed_pair_padding<_ToPad, true> {};
#endif // COMPRESSED_PAIR_REV == 1

#if COMPRESSED_PAIR_REV == 0
struct __value_init_tag {};
struct __default_init_tag {};

template <class _Tp, int _Idx,
          bool _CanBeEmptyBase =
              std::is_empty<_Tp>::value && !std::is_final<_Tp>::value>
struct __compressed_pair_elem {
  explicit __compressed_pair_elem(__default_init_tag) {}
  explicit __compressed_pair_elem(__value_init_tag) : __value_() {}

  explicit __compressed_pair_elem(_Tp __t) : __value_(__t) {}

  _Tp &__get() { return __value_; }

private:
  _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp {
  explicit __compressed_pair_elem(_Tp __t) : _Tp(__t) {}
  explicit __compressed_pair_elem(__default_init_tag) {}
  explicit __compressed_pair_elem(__value_init_tag) : _Tp() {}

  _Tp &__get() { return *this; }
};

template <class _T1, class _T2>
class __compressed_pair : private __compressed_pair_elem<_T1, 0>,
                          private __compressed_pair_elem<_T2, 1> {
public:
  using _Base1 = __compressed_pair_elem<_T1, 0>;
  using _Base2 = __compressed_pair_elem<_T2, 1>;

  explicit __compressed_pair(_T1 __t1, _T2 __t2) : _Base1(__t1), _Base2(__t2) {}
  explicit __compressed_pair()
      : _Base1(__value_init_tag()), _Base2(__value_init_tag()) {}

  template <class _U1, class _U2>
  explicit __compressed_pair(_U1 &&__t1, _U2 &&__t2)
      : _Base1(std::forward<_U1>(__t1)), _Base2(std::forward<_U2>(__t2)) {}

  _T1 &first() { return static_cast<_Base1 &>(*this).__get(); }
};
#elif COMPRESSED_PAIR_REV == 1 || COMPRESSED_PAIR_REV == 2
#define _LLDB_COMPRESSED_PAIR(T1, Initializer1, T2, Initializer2)              \
  [[__gnu__::__aligned__(                                                      \
      alignof(T2))]] _LLDB_NO_UNIQUE_ADDRESS T1 Initializer1;                  \
  _LLDB_NO_UNIQUE_ADDRESS __compressed_pair_padding<T1> __padding1_;           \
  _LLDB_NO_UNIQUE_ADDRESS T2 Initializer2;                                     \
  _LLDB_NO_UNIQUE_ADDRESS __compressed_pair_padding<T2> __padding2_;

#define _LLDB_COMPRESSED_TRIPLE(T1, Initializer1, T2, Initializer2, T3,        \
                                Initializer3)                                  \
  [[using __gnu__: __aligned__(alignof(T2)),                                   \
    __aligned__(alignof(T3))]] _LLDB_NO_UNIQUE_ADDRESS T1 Initializer1;        \
  _LLDB_NO_UNIQUE_ADDRESS __compressed_pair_padding<T1> __padding1_;           \
  _LLDB_NO_UNIQUE_ADDRESS T2 Initializer2;                                     \
  _LLDB_NO_UNIQUE_ADDRESS __compressed_pair_padding<T2> __padding2_;           \
  _LLDB_NO_UNIQUE_ADDRESS T3 Initializer3;                                     \
  _LLDB_NO_UNIQUE_ADDRESS __compressed_pair_padding<T3> __padding3_;
#elif COMPRESSED_PAIR_REV == 3
#define _LLDB_COMPRESSED_PAIR(T1, Name1, T2, Name2)                            \
  _LLDB_NO_UNIQUE_ADDRESS T1 Name1;                                            \
  _LLDB_NO_UNIQUE_ADDRESS T2 Name2

#define _LLDB_COMPRESSED_TRIPLE(T1, Name1, T2, Name2, T3, Name3)               \
  _LLDB_NO_UNIQUE_ADDRESS T1 Name1;                                            \
  _LLDB_NO_UNIQUE_ADDRESS T2 Name2;                                            \
  _LLDB_NO_UNIQUE_ADDRESS T3 Name3
#endif // COMPRESSED_PAIR_REV == 3
} // namespace __lldb
} // namespace std

#endif // _H
