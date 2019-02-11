// -*- C++ -*-
//===---------------------------- math.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_FENV_H
#define _LIBCPP_FENV_H


/*
    fenv.h synopsis

This entire header is C99 / C++0X

Macros:

    FE_DIVBYZERO
    FE_INEXACT
    FE_INVALID
    FE_OVERFLOW
    FE_UNDERFLOW
    FE_ALL_EXCEPT
    FE_DOWNWARD
    FE_TONEAREST
    FE_TOWARDZERO
    FE_UPWARD
    FE_DFL_ENV

Types:

    fenv_t
    fexcept_t

int feclearexcept(int excepts);
int fegetexceptflag(fexcept_t* flagp, int excepts);
int feraiseexcept(int excepts);
int fesetexceptflag(const fexcept_t* flagp, int excepts);
int fetestexcept(int excepts);
int fegetround();
int fesetround(int round);
int fegetenv(fenv_t* envp);
int feholdexcept(fenv_t* envp);
int fesetenv(const fenv_t* envp);
int feupdateenv(const fenv_t* envp);


*/

#include <__config>
#include_next <fenv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifdef __cplusplus

extern "C++" {

#ifdef feclearexcept
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_feclearexcept(int __excepts) {
  return feclearexcept(__excepts);
}
#undef feclearexcept
_LIBCPP_INLINE_VISIBILITY
inline int feclearexcept(int __excepts) {
  return ::__libcpp_feclearexcept(__excepts);
}
#endif // defined(feclearexcept)

#ifdef fegetexceptflag
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fegetexceptflag(fexcept_t* __out_ptr, int __excepts) {
  return fegetexceptflag(__out_ptr, __excepts);
}
#undef fegetexceptflag
_LIBCPP_INLINE_VISIBILITY
inline int fegetexceptflag(fexcept_t *__out_ptr, int __excepts) {
  return ::__libcpp_fegetexceptflag(__out_ptr, __excepts);
}
#endif // defined(fegetexceptflag)


#ifdef feraiseexcept
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_feraiseexcept(int __excepts) {
  return feraiseexcept(__excepts);
}
#undef feraiseexcept
_LIBCPP_INLINE_VISIBILITY
inline int feraiseexcept(int __excepts) {
  return ::__libcpp_feraiseexcept(__excepts);
}
#endif // defined(feraiseexcept)


#ifdef fesetexceptflag
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fesetexceptflag(const fexcept_t* __out_ptr, int __excepts) {
  return fesetexceptflag(__out_ptr, __excepts);
}
#undef fesetexceptflag
_LIBCPP_INLINE_VISIBILITY
inline int fesetexceptflag(const fexcept_t *__out_ptr, int __excepts) {
  return ::__libcpp_fesetexceptflag(__out_ptr, __excepts);
}
#endif // defined(fesetexceptflag)


#ifdef fetestexcept
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fetestexcept(int __excepts) {
  return fetestexcept(__excepts);
}
#undef fetestexcept
_LIBCPP_INLINE_VISIBILITY
inline int fetestexcept(int __excepts) {
  return ::__libcpp_fetestexcept(__excepts);
}
#endif // defined(fetestexcept)

#ifdef fegetround
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fegetround() {
  return fegetround();
}
#undef fegetround
_LIBCPP_INLINE_VISIBILITY
inline int fegetround() {
  return ::__libcpp_fegetround();
}
#endif // defined(fegetround)

#ifdef fesetround
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fesetround(int __round) {
  return fesetround(__round);
}
#undef fesetround
_LIBCPP_INLINE_VISIBILITY
inline int fesetround(int __round) {
  return ::__libcpp_fesetround(__round);
}
#endif // defined(fesetround)

#ifdef fegetenv
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fegetenv(fenv_t* __envp) {
  return fegetenv(__envp);
}
#undef fegetenv
_LIBCPP_INLINE_VISIBILITY
inline int fegetenv(fenv_t* __envp) {
  return ::__libcpp_fegetenv(__envp);
}
#endif // defined(fegetenv)

#ifdef feholdexcept
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_feholdexcept(fenv_t* __envp) {
  return feholdexcept(__envp);
}
#undef feholdexcept
_LIBCPP_INLINE_VISIBILITY
inline int feholdexcept(fenv_t* __envp) {
  return ::__libcpp_feholdexcept(__envp);
}
#endif // defined(feholdexcept)


#ifdef fesetenv
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_fesetenv(const fenv_t* __envp) {
  return fesetenv(__envp);
}
#undef fesetenv
_LIBCPP_INLINE_VISIBILITY
inline int fesetenv(const fenv_t* __envp) {
  return ::__libcpp_fesetenv(__envp);
}
#endif // defined(fesetenv)

#ifdef feupdateenv
_LIBCPP_INLINE_VISIBILITY
inline int __libcpp_feupdateenv(const fenv_t* __envp) {
  return feupdateenv(__envp);
}
#undef feupdateenv
_LIBCPP_INLINE_VISIBILITY
inline int feupdateenv(const fenv_t* __envp) {
  return ::__libcpp_feupdateenv(__envp);
}
#endif // defined(feupdateenv)

} // extern "C++"

#endif // defined(__cplusplus)

#endif // _LIBCPP_FENV_H
