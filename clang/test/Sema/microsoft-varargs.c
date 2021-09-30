// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only -x c %s -verify
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsyntax-only -x c %s -verify
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fsyntax-only -x c++ %s -verify
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fsyntax-only -x c++ %s -verify
// expected-no-diagnostics

#if defined _NO_CRT_STDIO_INLINE
#  undef _CRT_STDIO_INLINE
#  define _CRT_STDIO_INLINE
#elif !defined _CRT_STDIO_INLINE
#  define _CRT_STDIO_INLINE __inline
#endif

#ifndef _VA_LIST_DEFINED
#define _VA_LIST_DEFINED
typedef char *va_list;
#endif

#if !defined __cplusplus
// Workaround for /Zc:wchar_t
typedef  __WCHAR_TYPE__ wchar_t;
#endif

#if defined __cplusplus
#  define _ADDRESSOF(v) (&const_cast<char&>(reinterpret_cast<const volatile char&>(v)))
#else
#  define _ADDRESSOF(v) (&(v))
#endif

#if defined _M_ARM
#  define _VA_ALIGN      4
#  define _SLOTSIZEOF(t) ((sizeof(t) + _VA_ALIGN - 1) & ~(_VA_ALIGN - 1))
#  define _APALIGN(t,ap) (((va_list)0 - (ap)) & (__alignof(t) - 1))
#elif defined _M_ARM64
#  define _VA_ALIGN      8
#  define _SLOTSIZEOF(t) ((sizeof(t) + _VA_ALIGN - 1) & ~(_VA_ALIGN - 1))
#  define _APALIGN(t,ap) (((va_list)0 - (ap)) & (__alignof(t) - 1))
#endif

#if defined _M_ARM
void __cdecl __va_start(va_list*, ...);
#  if defined __cplusplus
#    define __crt_va_start_a(ap, v) ((void)(__va_start(&ap, _ADDRESSOF(v), _SLOTSIZEOF(v), _ADDRESSOF(v))))
#  else
#    define __crt_va_start_a(ap, v) ((void)(ap = (va_list)_ADDRESSOF(v) + _SLOTSIZEOF(v)))
#  endif

#  define __crt_va_arg(ap, t) (*(t*)((ap += _SLOTSIZEOF(t) + _APALIGN(t,ap)) - _SLOTSIZEOF(t)))
#  define __crt_va_end(ap)    ((void)(ap = (va_list)0))
#elif defined _M_ARM64
void __cdecl __va_start(va_list*, ...);
#  define __crt_va_start_a(ap,v) ((void)(__va_start(&ap, _ADDRESSOF(v), _SLOTSIZEOF(v), __alignof(v), _ADDRESSOF(v))))
#  define __crt_va_arg(ap, t)                                                   \
    ((sizeof(t) > (2 * sizeof(__int64)))                                        \
       ? **(t**)((ap += sizeof(__int64)) - sizeof(__int64))                     \
       : *(t*)((ap ++ _SLOTSIZEOF(t) + _APALIGN(t,ap)) - _SLOTSIZEOF(t)))
#  define __crt_va_end(ap)       ((void)(ap = (va_list)0))
#endif

#if defined __cplusplus
extern "C++" {
template <typename _T>
struct __vcrt_va_list_is_reference {
  enum : bool { __the_value = false };
};

template <typename _T>
struct __vcrt_va_list_is_reference<_T&> {
  enum : bool { __the_value = true };
};

template <typename _T>
struct __vcrt_va_list_is_reference<_T&&> {
  enum : bool { __the_value = true };
};

template <typename _T>
struct __vcrt_assert_va_start_is_not_reference {
  static_assert(!__vcrt_va_list_is_reference<_T>::__the_value,
                "va_start argument must not have reference type and must not be parenthesized");
};
}

#  define __crt_va_start(ap, x) ((void)(__vcrt_assert_va_start_is_not_reference<decltype(x)>(), __crt_va_start_a(ap, x)))
#else
#  define __crt_va_start(ap, x) __crt_va_start_a(ap, x)
#endif

/*_Check_return_opt_*/ _CRT_STDIO_INLINE int __cdecl
wprintf(/*_In_z_ _Printf_format_string_*/ wchar_t const * const _Format, ...) {
  int _Result;
  va_list _ArgList;
  __crt_va_start(_ArgList, _Format);
  // _Result = _vfwprintf_l(stdout, _Format, NULL, _ArgList);
  __crt_va_end(_ArgList);
  return _Result;
}
