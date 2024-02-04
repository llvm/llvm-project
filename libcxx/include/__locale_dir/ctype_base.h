//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_CTYPE_BASE_H
#define _LIBCPP___LOCALE_CTYPE_BASE_H

#include <__config>
#include <__type_traits/make_unsigned.h>
#include <cctype>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI ctype_base {
public:
#if defined(_LIBCPP_PROVIDES_DEFAULT_RUNE_TABLE)
  typedef unsigned long mask;
  static const mask space  = 1 << 0;
  static const mask print  = 1 << 1;
  static const mask cntrl  = 1 << 2;
  static const mask upper  = 1 << 3;
  static const mask lower  = 1 << 4;
  static const mask alpha  = 1 << 5;
  static const mask digit  = 1 << 6;
  static const mask punct  = 1 << 7;
  static const mask xdigit = 1 << 8;
  static const mask blank  = 1 << 9;
#  if defined(__BIONIC__)
  // Historically this was a part of regex_traits rather than ctype_base. The
  // historical value of the constant is preserved for ABI compatibility.
  static const mask __regex_word = 0x8000;
#  else
  static const mask __regex_word = 1 << 10;
#  endif // defined(__BIONIC__)
#elif defined(__GLIBC__)
  typedef unsigned short mask;
  static const mask space  = _ISspace;
  static const mask print  = _ISprint;
  static const mask cntrl  = _IScntrl;
  static const mask upper  = _ISupper;
  static const mask lower  = _ISlower;
  static const mask alpha  = _ISalpha;
  static const mask digit  = _ISdigit;
  static const mask punct  = _ISpunct;
  static const mask xdigit = _ISxdigit;
  static const mask blank  = _ISblank;
#  if defined(__mips__) || defined(_LIBCPP_BIG_ENDIAN)
  static const mask __regex_word = static_cast<mask>(_ISbit(15));
#  else
  static const mask __regex_word = 0x80;
#  endif
#elif defined(_LIBCPP_MSVCRT_LIKE)
  typedef unsigned short mask;
  static const mask space        = _SPACE;
  static const mask print        = _BLANK | _PUNCT | _ALPHA | _DIGIT;
  static const mask cntrl        = _CONTROL;
  static const mask upper        = _UPPER;
  static const mask lower        = _LOWER;
  static const mask alpha        = _ALPHA;
  static const mask digit        = _DIGIT;
  static const mask punct        = _PUNCT;
  static const mask xdigit       = _HEX;
  static const mask blank        = _BLANK;
  static const mask __regex_word = 0x4000; // 0x8000 and 0x0100 and 0x00ff are used
#  define _LIBCPP_CTYPE_MASK_IS_COMPOSITE_PRINT
#  define _LIBCPP_CTYPE_MASK_IS_COMPOSITE_ALPHA
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__EMSCRIPTEN__) || defined(__NetBSD__)
#  ifdef __APPLE__
  typedef __uint32_t mask;
#  elif defined(__FreeBSD__)
  typedef unsigned long mask;
#  elif defined(__EMSCRIPTEN__) || defined(__NetBSD__)
  typedef unsigned short mask;
#  endif
  static const mask space  = _CTYPE_S;
  static const mask print  = _CTYPE_R;
  static const mask cntrl  = _CTYPE_C;
  static const mask upper  = _CTYPE_U;
  static const mask lower  = _CTYPE_L;
  static const mask alpha  = _CTYPE_A;
  static const mask digit  = _CTYPE_D;
  static const mask punct  = _CTYPE_P;
  static const mask xdigit = _CTYPE_X;

#  if defined(__NetBSD__)
  static const mask blank = _CTYPE_BL;
  // NetBSD defines classes up to 0x2000
  // see sys/ctype_bits.h, _CTYPE_Q
  static const mask __regex_word = 0x8000;
#  else
  static const mask blank        = _CTYPE_B;
  static const mask __regex_word = 0x80;
#  endif
#elif defined(_AIX)
  typedef unsigned int mask;
  static const mask space        = _ISSPACE;
  static const mask print        = _ISPRINT;
  static const mask cntrl        = _ISCNTRL;
  static const mask upper        = _ISUPPER;
  static const mask lower        = _ISLOWER;
  static const mask alpha        = _ISALPHA;
  static const mask digit        = _ISDIGIT;
  static const mask punct        = _ISPUNCT;
  static const mask xdigit       = _ISXDIGIT;
  static const mask blank        = _ISBLANK;
  static const mask __regex_word = 0x8000;
#elif defined(_NEWLIB_VERSION)
  // Same type as Newlib's _ctype_ array in newlib/libc/include/ctype.h.
  typedef char mask;
  // In case char is signed, static_cast is needed to avoid warning on
  // positive value becomming negative.
  static const mask space  = static_cast<mask>(_S);
  static const mask print  = static_cast<mask>(_P | _U | _L | _N | _B);
  static const mask cntrl  = static_cast<mask>(_C);
  static const mask upper  = static_cast<mask>(_U);
  static const mask lower  = static_cast<mask>(_L);
  static const mask alpha  = static_cast<mask>(_U | _L);
  static const mask digit  = static_cast<mask>(_N);
  static const mask punct  = static_cast<mask>(_P);
  static const mask xdigit = static_cast<mask>(_X | _N);
  static const mask blank  = static_cast<mask>(_B);
  // mask is already fully saturated, use a different type in regex_type_traits.
  static const unsigned short __regex_word = 0x100;
#  define _LIBCPP_CTYPE_MASK_IS_COMPOSITE_PRINT
#  define _LIBCPP_CTYPE_MASK_IS_COMPOSITE_ALPHA
#  define _LIBCPP_CTYPE_MASK_IS_COMPOSITE_XDIGIT
#elif defined(__MVS__)
#  if defined(__NATIVE_ASCII_F)
  typedef unsigned int mask;
  static const mask space  = _ISSPACE_A;
  static const mask print  = _ISPRINT_A;
  static const mask cntrl  = _ISCNTRL_A;
  static const mask upper  = _ISUPPER_A;
  static const mask lower  = _ISLOWER_A;
  static const mask alpha  = _ISALPHA_A;
  static const mask digit  = _ISDIGIT_A;
  static const mask punct  = _ISPUNCT_A;
  static const mask xdigit = _ISXDIGIT_A;
  static const mask blank  = _ISBLANK_A;
#  else
  typedef unsigned short mask;
  static const mask space  = __ISSPACE;
  static const mask print  = __ISPRINT;
  static const mask cntrl  = __ISCNTRL;
  static const mask upper  = __ISUPPER;
  static const mask lower  = __ISLOWER;
  static const mask alpha  = __ISALPHA;
  static const mask digit  = __ISDIGIT;
  static const mask punct  = __ISPUNCT;
  static const mask xdigit = __ISXDIGIT;
  static const mask blank  = __ISBLANK;
#  endif
  static const mask __regex_word = 0x8000;
#else
#  error unknown rune table for this platform -- do you mean to define _LIBCPP_PROVIDES_DEFAULT_RUNE_TABLE?
#endif
  static const mask alnum = alpha | digit;
  static const mask graph = alnum | punct;

  _LIBCPP_HIDE_FROM_ABI ctype_base() {}

  static_assert((__regex_word & ~(std::make_unsigned<mask>::type)(space | print | cntrl | upper | lower | alpha |
                                                                  digit | punct | xdigit | blank)) == __regex_word,
                "__regex_word can't overlap other bits");
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CTYPE_BASE_H
