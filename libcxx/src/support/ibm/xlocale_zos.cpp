//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/ibm_xlocale.h"

#include <__assert>
#include <__locale_dir/support/zos.h>
#include <errno.h>

_LIBCPP_BEGIN_NAMESPACE_STD

#define CategoryList(pair, sep)                                                                                        \
  pair(LC_COLLATE, lc_collate) sep pair(LC_CTYPE, lc_ctype)                                                            \
  sep pair(LC_MONETARY, lc_monetary)                                                                                   \
  sep pair(LC_NUMERIC, lc_numeric)                                                                                     \
  sep pair(LC_TIME, lc_time)                                                                                           \
  sep pair(LC_MESSAGES, lc_messages)

// check ids and masks agree
#define check_ids_and_masks_agree(id, _)                                                                               \
  static_assert((1 << id) == id##_MASK, "id and mask do not agree for " #id);                                          \
  static_assert((1 << id) == _CATMASK(id), "mask does not have expected value for " #id);
CategoryList(check_ids_and_masks_agree, )
#undef check_ids_and_masks_agree

// check that LC_ALL_MASK is defined as expected
#define get_mask(id, _) id##_MASK
    static_assert(
        LC_ALL_MASK == (CategoryList(get_mask, |)),
        "LC_ALL_MASK does not have the expected value.  Check that its definition includes all supported categories");
#undef get_mask

// initialize c_locale
#define init_clocale(id, locale_member) "C",
static locale_struct c_locale = {LC_ALL_MASK, CategoryList(init_clocale, )};
#undef init_clocale

static locale_t current_locale = _LIBCPP_LC_GLOBAL_LOCALE;

locale_t __c_locale() { return &c_locale; }

// locale
locale_t newlocale(int category_mask, const char* locale, locale_t base) {
  // start with some basic checks
  if (!locale) {
    errno = EINVAL;
    return (locale_t)0;
  }
  if (category_mask & ~LC_ALL_MASK) {
    // then there are bits in category_mask that does not correspond
    // to a valid category
    errno = EINVAL;
    return (locale_t)0;
  }

  locale_t new_loc          = new locale_struct;
  int num_locales_not_found = 0;

  if (base && base != _LIBCPP_LC_GLOBAL_LOCALE)
    *new_loc = *base;

  auto set_for_category = [&](int id, int mask, std::string& setting) {
    const char* setting_to_apply = nullptr;

    if (category_mask & mask)
      setting_to_apply = locale;
    else if (!base)
      setting_to_apply = "C";

    if (setting_to_apply) {
      // setlocale takes the id, not the mask
      const char* saved_setting = setlocale(id, nullptr);
      if (setlocale(id, setting_to_apply)) {
        // then setting_to_apply is valid for this category
        // restore the saved setting
        setlocale(id, saved_setting);

        new_loc->category_mask |= mask;
        setting = setting_to_apply;
      } else {
        // unknown locale for this category
        num_locales_not_found++;
      }
    }
  };

  // now overlay values
#define Set(id, locale_member) set_for_category(id, id##_MASK, new_loc->locale_member)
  CategoryList(Set, ;);
#undef Set

  if (num_locales_not_found != 0) {
    delete new_loc;
    errno   = ENOENT;
    new_loc = (locale_t)0;
  }

  return new_loc;
}

void freelocale(locale_t locobj) {
  if (locobj != nullptr && locobj != &c_locale && locobj != _LIBCPP_LC_GLOBAL_LOCALE)
    delete locobj;
}

locale_t uselocale(locale_t new_loc) {
  locale_t prev_loc = current_locale;

  if (new_loc == _LIBCPP_LC_GLOBAL_LOCALE) {
    current_locale = _LIBCPP_LC_GLOBAL_LOCALE;
  } else if (new_loc != nullptr) {
    locale_struct saved_locale;
    saved_locale.category_mask = 0;

    auto apply_category = [&](int id, int mask, std::string& setting, std::string& save_setting) -> bool {
      if (new_loc->category_mask & mask) {
        const char* old_setting = setlocale(id, setting.c_str());
        if (old_setting) {
          // we were able to set for this category.  Save the old setting
          // in case a subsequent category fails, and we need to restore
          // all earlier settings.
          saved_locale.category_mask |= mask;
          save_setting = old_setting;
          return true;
        }
        return false;
      }
      return true;
    };

#define Apply(id, locale_member) apply_category(id, id##_MASK, new_loc->locale_member, saved_locale.locale_member)
    bool is_ok = CategoryList(Apply, &&);
#undef Apply

    if (!is_ok) {
      auto restore = [&](int id, int mask, std::string& setting) {
        if (saved_locale.category_mask & mask)
          setlocale(id, setting.c_str());
      };
#define Restore(id, locale_member) restore(id, id##_MASK, saved_locale.locale_member);
      CategoryList(Restore, );
#undef Restore
      errno = EINVAL;
      return nullptr;
    }
    current_locale = new_loc;
  }

  return prev_loc;
}

int isdigit_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isdigit(__c);
}

int isxdigit_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isxdigit(__c);
}

namespace __locale {
namespace __ibm {
int isalnum_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isalnum(__c);
}

int isalpha_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isalpha(__c);
}

int isblank_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isblank(__c);
}

int iscntrl_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iscntrl(__c);
}

int isgraph_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isgraph(__c);
}

int islower_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::islower(__c);
}

int isprint_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isprint(__c);
}

int ispunct_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::ispunct(__c);
}

int isspace_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isspace(__c);
}

int isupper_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::isupper(__c);
}

int iswalnum_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswalnum(__c);
}

int iswalpha_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswalpha(__c);
}

int iswblank_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswblank(__c);
}

int iswcntrl_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswcntrl(__c);
}

int iswdigit_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswdigit(__c);
}

int iswgraph_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswgraph(__c);
}

int iswlower_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswlower(__c);
}

int iswprint_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswprint(__c);
}

int iswpunct_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswpunct(__c);
}

int iswspace_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswspace(__c);
}

int iswupper_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswupper(__c);
}

int iswxdigit_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswxdigit(__c);
}

int toupper_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::toupper(__c);
}

int tolower_l(int __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::tolower(__c);
}

wint_t towupper_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::towupper(__c);
}

wint_t towlower_l(wint_t __c, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::towlower(__c);
}

int strcoll_l(const char* __s1, const char* __s2, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::strcoll(__s1, __s2);
}

size_t strxfrm_l(char* __dest, const char* __src, size_t __n, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::strxfrm(__dest, __src, __n);
}

size_t strftime_l(char* __s, size_t __max, const char* __format, const struct tm* __tm, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::strftime(__s, __max, __format, __tm);
}

int wcscoll_l(const wchar_t* __ws1, const wchar_t* __ws2, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::wcscoll(__ws1, __ws2);
}

size_t wcsxfrm_l(wchar_t* __dest, const wchar_t* __src, size_t __n, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::wcsxfrm(__dest, __src, __n);
}

int iswctype_l(wint_t __c, wctype_t __type, locale_t __l) {
  __locale::__locale_guard __newloc(__l);
  return ::iswctype(__c, __type);
}
} // namespace __ibm
} // namespace __locale
_LIBCPP_END_NAMESPACE_STD
