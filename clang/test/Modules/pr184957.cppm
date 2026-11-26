// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: mkdir %t/tmp
//
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std %t/wrap.std.tt.cppm -emit-reduced-module-interface -o %t/wrap.std.tt.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std %t/wrap.std.vec.cppm -emit-reduced-module-interface -o %t/wrap.std.vec.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std %t/not_std.cppm -emit-reduced-module-interface -o %t/not_std.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std %t/wrap.std.tt2.cppm -emit-reduced-module-interface -o %t/wrap.std.tt2.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std -fmodule-file=wrap.std.vec=%t/wrap.std.vec.pcm %t/wrap.std.vec.reexport.cppm -emit-reduced-module-interface -o %t/wrap.std.vec.reexport.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std -fmodule-file=wrap.std.tt=%t/wrap.std.tt.pcm -fmodule-file=wrap.std.vec=%t/wrap.std.vec.pcm -fmodule-file=wrap.std.vec.reexport=%t/wrap.std.vec.reexport.pcm -fmodule-file=wrap.std.tt2=%t/wrap.std.tt2.pcm -fmodule-file=not_std=%t/not_std.pcm %t/k.repro_dep.cppm -emit-reduced-module-interface -o %t/k.repro_dep.pcm
// RUN: %clang_cc1 -std=c++23 -nostdinc++ -I %t/std -fmodule-file=wrap.std.tt=%t/wrap.std.tt.pcm -fmodule-file=wrap.std.vec=%t/wrap.std.vec.pcm -fmodule-file=wrap.std.tt2=%t/wrap.std.tt2.pcm -fmodule-file=wrap.std.vec.reexport=%t/wrap.std.vec.reexport.pcm -fmodule-file=not_std=%t/not_std.pcm -fmodule-file=k.repro_dep=%t/k.repro_dep.pcm %t/k.repro.cxx -fsyntax-only -verify

//--- std/allocator.h
#ifndef _LIBCPP___MEMORY_ALLOCATOR_H
#define _LIBCPP___MEMORY_ALLOCATOR_H

#include <size_t.h>

namespace std {

enum align_val_t { __zero = 0, __max = (size_t)-1 };

template <class _Tp>
inline _Tp*
__libcpp_allocate(size_t __n, [[__maybe_unused__]] size_t __align = 8) {
  size_t __size = static_cast<size_t>(__n) * sizeof(_Tp);
  return static_cast<_Tp*>(__builtin_operator_new(__size, static_cast<align_val_t>(__align)));
}

template <class _Tp>
class allocator
{
public:
  typedef _Tp value_type;

  [[__nodiscard__]] _Tp* allocate(size_t __n) {
    if (__builtin_is_constant_evaluated()) {
      return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
    } else {
      return std::__libcpp_allocate<_Tp>(size_t(__n));
    }
  }
};

template <class _Tp, class _Up>
inline bool
operator==(const allocator<_Tp>&, const allocator<_Up>&) noexcept {
  return true;
}

}

#endif // _LIBCPP___MEMORY_ALLOCATOR_H

//--- std/sys_wchar.h
#ifndef _WCHAR_H
#define _WCHAR_H 1

#ifndef __FILE_defined
  #define __FILE_defined 1

  struct _IO_FILE;

  typedef struct _IO_FILE FILE;
#endif

#endif /* wchar.h  */

//--- std/char_traits.h
#ifndef _LIBCPP___STRING_CHAR_TRAITS_H
#define _LIBCPP___STRING_CHAR_TRAITS_H

namespace std {

template <class _Tp>
inline size_t constexpr __constexpr_strlen(const _Tp* __str) noexcept {
  if (__builtin_is_constant_evaluated()) {
    size_t __i = 0;
    for (; __str[__i] != '\0'; ++__i)
      ;
    return __i;
  }
  return 0;
}

template <class _CharT>
struct char_traits;

template <>
struct char_traits<char> {
  using char_type  = char;
  
  [[__nodiscard__]] static inline size_t constexpr
  length(const char_type* __s) noexcept {
    return std::__constexpr_strlen(__s);
  }
};

}

#endif // _LIBCPP___STRING_CHAR_TRAITS_H

//--- std/type_traits
#ifndef _LIBCPP_TYPE_TRAITS
#define _LIBCPP_TYPE_TRAITS

namespace std
{}

#endif // _LIBCPP_TYPE_TRAITS

//--- std/ptrdiff_t.h
#ifndef _LIBCPP___CSTDDEF_PTRDIFF_T_H
#define _LIBCPP___CSTDDEF_PTRDIFF_T_H

namespace std {

using ptrdiff_t = decltype(static_cast<int*>(nullptr) - static_cast<int*>(nullptr));

}

#endif // _LIBCPP___CSTDDEF_PTRDIFF_T_H

//--- std/size_t.h
#ifndef _LIBCPP___CSTDDEF_SIZE_T_H
#define _LIBCPP___CSTDDEF_SIZE_T_H

namespace std {

using size_t = decltype(sizeof(int));

}

#endif // _LIBCPP___CSTDDEF_SIZE_T_H

//--- wrap.std.tt.cppm
module;

#include <type_traits>

export module wrap.std.tt;

//--- wrap.std.vec.cppm
module;

#include <allocator.h>
#include <sys_wchar.h>

export module wrap.std.vec;

//--- not_std.cppm
module;

#include <allocator.h>


export module not_std;

namespace std {
  template <class _Tp, class _Allocator>
  class __split_buffer {
  public:
    __split_buffer() {
      _Allocator a;
      (void)a.allocate(1);
    }
  };
}

export namespace std {
  template <typename T>
  using problem = std::__split_buffer<T, std::allocator<T>>;
}

export using ::operator new;

//--- wrap.std.tt2.cppm
module;

#include <type_traits>

export module wrap.std.tt2;

//--- wrap.std.vec.reexport.cppm
export module wrap.std.vec.reexport;

export import wrap.std.vec;

//--- k.repro_dep.cppm
module;

#include <allocator.h>
#include <sys_wchar.h>
#include <char_traits.h>

export module k.repro_dep;

import wrap.std.tt;
import wrap.std.vec.reexport;
import wrap.std.vec;
import wrap.std.tt2;

import not_std;

using XYZ = std::char_traits<char>;

//--- k.repro.cxx
// expected-no-diagnostics
import not_std;
import k.repro_dep;

auto f() -> void
{
  std::problem< int > x;
}
