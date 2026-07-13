//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-wide-characters

// towctrans and wctrans were added in Android API 26.
// TODO: Switch from UNSUPPORTED to XFAIL once the Android CI Docker sysroot is
// updated.
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && target={{.+}}-android{{(eabi)?(21|22|23|24|25)}}

// <wctype.h>

#include <wctype.h>

#include "test_macros.h"

#ifndef WEOF
#error WEOF not defined
#endif

#ifdef iswalnum
#error iswalnum defined
#endif

#ifdef iswalpha
#error iswalpha defined
#endif

#ifdef iswblank
#error iswblank defined
#endif

#ifdef iswcntrl
#error iswcntrl defined
#endif

#ifdef iswdigit
#error iswdigit defined
#endif

#ifdef iswgraph
#error iswgraph defined
#endif

#ifdef iswlower
#error iswlower defined
#endif

#ifdef iswprint
#error iswprint defined
#endif

#ifdef iswpunct
#error iswpunct defined
#endif

#ifdef iswspace
#error iswspace defined
#endif

#ifdef iswupper
#error iswupper defined
#endif

#ifdef iswxdigit
#error iswxdigit defined
#endif

#ifdef iswctype
#error iswctype defined
#endif

#ifdef wctype
#error wctype defined
#endif

#ifdef towlower
#error towlower defined
#endif

#ifdef towupper
#error towupper defined
#endif

#ifdef towctrans
#error towctrans defined
#endif

#ifdef wctrans
#error wctrans defined
#endif

wint_t w = 0;
wctrans_t wctr = 0;
wctype_t wct = 0;
ASSERT_SAME_TYPE(int,       decltype(iswalnum(w)));
ASSERT_SAME_TYPE(int,       decltype(iswalpha(w)));
ASSERT_SAME_TYPE(int,       decltype(iswblank(w)));
ASSERT_SAME_TYPE(int,       decltype(iswcntrl(w)));
ASSERT_SAME_TYPE(int,       decltype(iswdigit(w)));
ASSERT_SAME_TYPE(int,       decltype(iswgraph(w)));
ASSERT_SAME_TYPE(int,       decltype(iswlower(w)));
ASSERT_SAME_TYPE(int,       decltype(iswprint(w)));
ASSERT_SAME_TYPE(int,       decltype(iswpunct(w)));
ASSERT_SAME_TYPE(int,       decltype(iswspace(w)));
ASSERT_SAME_TYPE(int,       decltype(iswupper(w)));
ASSERT_SAME_TYPE(int,       decltype(iswxdigit(w)));
ASSERT_SAME_TYPE(int,       decltype(iswctype(w, wct)));
ASSERT_SAME_TYPE(wctype_t,  decltype(wctype("")));
ASSERT_SAME_TYPE(wint_t,    decltype(towlower(w)));
ASSERT_SAME_TYPE(wint_t,    decltype(towupper(w)));
ASSERT_SAME_TYPE(wint_t,    decltype(towctrans(w, wctr)));
ASSERT_SAME_TYPE(wctrans_t, decltype(wctrans("")));
