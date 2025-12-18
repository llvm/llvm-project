//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-zos{{.*}}

// Test that version of the POSIX functions provided outside of libc++ don't
// cause compilation errors.

struct locale_t {};
locale_t newlocale(int category_mask, const char* locale, locale_t base);
void freelocale(locale_t locobj);
locale_t uselocale(locale_t newloc);

#ifdef _LP64
typedef unsigned long size_t;
#else
typedef unsigned int size_t;
#endif
typedef short mbstate_t;
size_t mbsnrtowcs(wchar_t*, const char**, size_t, size_t, mbstate_t*);
size_t wcsnrtombs(char*, const wchar_t**, size_t, size_t, mbstate_t*);

#include <locale>
