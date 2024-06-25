// Test case reduced from an experimental std modules implementation.
// Tests that the compiler don't emit confusing error about the ambiguous ctor
// about std::pair.
//
// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/string.cppm -I%t -emit-module-interface -o %t/std-string.pcm
// RUN: %clang_cc1 -std=c++20 %t/algorithm.cppm -I%t -emit-module-interface -o %t/std-algorithm.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cppm -I%t -fprebuilt-module-path=%t -emit-module-interface -verify -o %t/Use.pcm

// Test again with reduced BMI.
// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/string.cppm -I%t -emit-reduced-module-interface -o %t/std-string.pcm
// RUN: %clang_cc1 -std=c++20 %t/algorithm.cppm -I%t -emit-reduced-module-interface -o %t/std-algorithm.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cppm -I%t -fprebuilt-module-path=%t -emit-reduced-module-interface -verify -o %t/Use.pcm

//--- Use.cppm
// expected-no-diagnostics
module;
#include "config.h"
# 3 "pair-unambiguous-ctor.cppm" 1 3
export module std:M;
# 3 "pair-unambiguous-ctor.cppm" 2 3
import :string;
import :algorithm;

auto check() {
    return std::string();
}

//--- string.cppm
module;
#include "string.h"
# 28 "pair-unambiguous-ctor.cppm" 1 3
export module std:string;
export namespace std {
    using std::string;
}
# 28 "pair-unambiguous-ctor.cppm" 2 3

//--- algorithm.cppm
module;
#include "algorithm.h"
# 38 "pair-unambiguous-ctor.cppm" 1 3
export module std:algorithm;
# 38 "pair-unambiguous-ctor.cppm" 2 3

//--- pair.h
namespace std __attribute__ ((__visibility__ ("default")))
{ 
  typedef long unsigned int size_t;
  typedef long int ptrdiff_t;

  typedef decltype(nullptr) nullptr_t;

  template<typename _Tp, _Tp __v>
    struct integral_constant
    {
      static constexpr _Tp value = __v;
      typedef _Tp value_type;
      typedef integral_constant<_Tp, __v> type;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
    };

  template<typename _Tp, _Tp __v>
    constexpr _Tp integral_constant<_Tp, __v>::value;

  typedef integral_constant<bool, true> true_type;
  typedef integral_constant<bool, false> false_type;

  template<bool __v>
    using __bool_constant = integral_constant<bool, __v>;


  template<bool, typename, typename>
    struct conditional;

  template<bool _Cond, typename _Iftrue, typename _Iffalse>
    struct conditional
    { typedef _Iftrue type; };

  template<typename _Iftrue, typename _Iffalse>
    struct conditional<false, _Iftrue, _Iffalse>
    { typedef _Iffalse type; };


  template<bool, typename _Tp = void>
    struct enable_if
    { };


  template<typename _Tp>
    struct enable_if<true, _Tp>
    { typedef _Tp type; };

  template<typename _Tp, typename... _Args>
    struct __is_constructible_impl
    : public __bool_constant<__is_constructible(_Tp, _Args...)>
    { };


  template<typename _Tp, typename... _Args>
    struct is_constructible
      : public __is_constructible_impl<_Tp, _Args...>
    {};

  template<typename>
    struct __is_void_helper
    : public false_type { };

  template<>
    struct __is_void_helper<void>
    : public true_type { };

  template<typename _Tp>
    struct is_void
    : public __is_void_helper<_Tp>::type
    { };

  template<typename...>
    class tuple;

  template<std::size_t...>
    struct _Index_tuple;

  template <bool, typename _T1, typename _T2>
    struct _PCC
    {
      template <typename _U1, typename _U2>
      static constexpr bool _ConstructiblePair()
      {
 return is_constructible<_T1, const _U1&>::value;
      }

  };

  template<typename _T1, typename _T2>
    struct pair
    {
      typedef _T1 first_type;
      typedef _T2 second_type;

      _T1 first;
      _T2 second;

      using _PCCP = _PCC<true, _T1, _T2>;

      template<typename _U1 = _T1, typename _U2=_T2, typename
        enable_if<_PCCP::template
      _ConstructiblePair<_U1, _U2>(),
                         bool>::type=true>
      constexpr pair(const _T1& __a, const _T2& __b)
      : first(__a), second(__b) { }

      constexpr pair&
      operator=(typename conditional<
         is_constructible<_T2>::value,
  const pair&, nullptr_t>::type __p)
      {
 first = __p.first;
 second = __p.second;
 return *this;
      }

    private:
      template<typename... _Args1, std::size_t... _Indexes1,
               typename... _Args2, std::size_t... _Indexes2>
      constexpr
      pair(tuple<_Args1...>&, tuple<_Args2...>&,
           _Index_tuple<_Indexes1...>, _Index_tuple<_Indexes2...>);

    };

  template<typename _T1, typename _T2> pair(_T1, _T2) -> pair<_T1, _T2>;
}

//--- string.h
#include "pair.h"

namespace std __attribute__ ((__visibility__ ("default")))
{
  class __undefined;

  template<typename _Tp>
    using __make_not_void
      = typename conditional<is_void<_Tp>::value, __undefined, _Tp>::type;

  template <typename Ptr>
  struct pointer_traits {};
  
  template<typename _Tp>
    struct pointer_traits<_Tp*>
    {

      typedef _Tp* pointer;

      typedef _Tp element_type;

      static constexpr pointer
      pointer_to(__make_not_void<element_type>& __r) noexcept
      { return __builtin_addressof(__r); }
    };

  template<typename _Tp>
    class allocator;

  template<typename _Alloc>
    struct allocator_traits;

  template<typename _Tp>
    struct allocator_traits<allocator<_Tp>>
    {
      using pointer = _Tp*;
    };

  template<typename _Alloc>
  struct __alloc_traits
  : std::allocator_traits<_Alloc>
  {
    typedef std::allocator_traits<_Alloc> _Base_type;
    typedef typename _Base_type::pointer pointer;
  };

  template<class _CharT>
    struct char_traits;

  template<typename _CharT, typename _Traits = char_traits<_CharT>,
           typename _Alloc = allocator<_CharT> >
    class basic_string
    {
      typedef std::__alloc_traits<_Alloc> _Alloc_traits;

    public:
      typedef typename _Alloc_traits::pointer pointer;

    private:
      pointer _M_dataplus;
      _CharT _M_local_buf[16];

      pointer
      _M_local_data()
      {
        return std::pointer_traits<pointer>::pointer_to(*_M_local_buf);
      }
    public:
      basic_string()
      : _M_dataplus(_M_local_data())
      { }

    };

    typedef basic_string<char> string;
}

//--- algorithm.h
#include "pair.h"
namespace std {
    struct _Power2_rehash_policy
  {
    std::pair<bool, std::size_t>
    _M_need_rehash(std::size_t __n_bkt, std::size_t __n_elt,
     std::size_t __n_ins) noexcept
    {
        return { false, 0 };
    }
  };
}

//--- config.h
namespace std
{
  typedef __SIZE_TYPE__ 	size_t;
}

