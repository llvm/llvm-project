
#ifdef __sun__
      
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <dlfcn.h>
#include <locale.h>
#include <limits.h>
#include <assert.h>
#include <sys/localedef.h>
#include "xlocale.h"

static _LC_locale_t *__C_locale;

#define FIX_LOCALE(l) l = (l == 0) ? __C_locale : l

#include "mbsnrtowcs.inc"
#include "wcsnrtombs.inc"
      
size_t __mb_cur_max(locale_t __l) {
  FIX_LOCALE(__l);
  return (__l->lc_ctype->cmapp->cm_mb_cur_max);
}

wint_t btowc_l(int __c, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->btowc(__l->lc_ctype->cmapp, __c);
}

int wctob_l(wint_t __c, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->wctob(__l->lc_ctype->cmapp, __c);
}

size_t wcrtomb_l(char *__s, wchar_t __wc, mbstate_t *__ps, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->wcrtomb(__l->lc_ctype->cmapp,
      __s, __wc, __ps);
}

size_t mbrtowc_l(wchar_t *__pwc, const char *__s, size_t __n,
                 mbstate_t *__ps, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->mbrtowc(__l->lc_ctype->cmapp,
      __pwc, __s, __n, __ps);
}

int mbtowc_l(wchar_t *__pwc, const char *__pmb, size_t __max, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->mbtowc(__l->lc_ctype->cmapp,
      __pwc, __pmb, __max);
} 
  
size_t mbrlen_l(const char *__s, size_t __n, mbstate_t *__ps, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->mbrlen(__l->lc_ctype->cmapp, __s,
    __n, __ps);
}

struct lconv *localeconv_l(locale_t __l) {
  FIX_LOCALE(__l);
  return __l->core.user_api->localeconv(__l);
} 
  
size_t mbsrtowcs_l(wchar_t *__dest, const char **__src, size_t __len,
                   mbstate_t *__ps, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->cmapp->core.user_api->mbsrtowcs(__l->lc_ctype->cmapp,
      __dest, __src, __len, __ps);
}

int wcscoll_l(const wchar_t *__s1, const wchar_t *__s2, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_collate->core.user_api->wcscoll(__l->lc_collate,
      __s1, __s2);
}

int strcoll_l(const char *__s1, const char *__s2, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_collate->core.user_api->strcoll(__l->lc_collate,
      __s1, __s2);
}

size_t strxfrm_l(char *__s1, const char *__s2, size_t __n, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_collate->core.user_api->strxfrm(__l->lc_collate,
      __s1, __s2, __n);
}
size_t strftime_l(char *__s, size_t __size, const char *__fmt, const struct tm
    *__tm, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_time->core.user_api->strftime(__l->lc_time,
      __s, __size, __fmt, __tm);
}

size_t wcsxfrm_l(wchar_t *__ws1, const wchar_t *__ws2, size_t __n,
    locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_collate->core.user_api->wcsxfrm(__l->lc_collate,
      __ws1, __ws2, __n);
}

#define LOCALE_ISCTYPE(ctype, m) \
  int is##ctype##_l(int __c, locale_t __l) { \
    if ((__c < 0) || (__c > 255)) return 0;\
    FIX_LOCALE(__l);\
    return __l->lc_ctype->mask[__c] & m;\
  }\
  int isw##ctype##_l(wchar_t __c, locale_t __l) { \
    FIX_LOCALE(__l);\
    return __l->lc_ctype->core.user_api->iswctype(__l->lc_ctype, __c, m);\
  }

LOCALE_ISCTYPE(alnum, _ISALNUM)
LOCALE_ISCTYPE(alpha, _ISALPHA)
LOCALE_ISCTYPE(blank, _ISALPHA)
LOCALE_ISCTYPE(cntrl, _ISCNTRL)
LOCALE_ISCTYPE(digit, _ISDIGIT)
LOCALE_ISCTYPE(graph, _ISGRAPH)
LOCALE_ISCTYPE(lower, _ISLOWER)
LOCALE_ISCTYPE(print, _ISPRINT)
LOCALE_ISCTYPE(punct, _ISPUNCT)
LOCALE_ISCTYPE(space, _ISSPACE)
LOCALE_ISCTYPE(upper, _ISUPPER)
LOCALE_ISCTYPE(xdigit, _ISXDIGIT)

int iswctype_l(wint_t __c, wctype_t __m, locale_t __l) {
    FIX_LOCALE(__l);\
    return __l->lc_ctype->core.user_api->iswctype(__l->lc_ctype, __c, __m);\
}

int toupper_l(int __c, locale_t __l) {
  FIX_LOCALE(__l);
    if ((__c < 0) || (__c > __l->lc_ctype->max_upper)) return __c;
  return __l->lc_ctype->upper[__c];
}
int tolower_l(int __c, locale_t __l) {
  FIX_LOCALE(__l);
  if ((__c < 0) || (__c > __l->lc_ctype->max_lower)) return __c;
  return __l->lc_ctype->lower[__c];
}
wint_t towupper_l(wint_t __c, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->core.user_api->towupper(__l->lc_ctype, __c);
}
wint_t towlower_l(wint_t __c, locale_t __l) {
  FIX_LOCALE(__l);
  return __l->lc_ctype->core.user_api->towlower(__l->lc_ctype, __c);
}

// FIXME: This disregards the locale, which is Very Wrong
#define vsnprintf_l(__s, __n, __l, __format, __va)  \
    vsnprintf(__s, __n, __format, __va) 

int sprintf_l(char *__s, locale_t __l, const char *__format, ...) {
  va_list __va;
  va_start(__va, __format);
  int __res = vsnprintf_l(__s, SIZE_MAX, __l, __format, __va);
  va_end(__va);
  return __res;
}

int snprintf_l(char *__s, size_t __n, locale_t __l, const char *__format, ...)
{
  va_list __va;
  va_start(__va, __format);
  int __res = vsnprintf_l(__s, __n , __l, __format, __va);
  va_end(__va);
  return __res;
}

int asprintf_l(char **__s, locale_t __l, const char *__format, ...) {
  va_list __va;
  va_start(__va, __format);
  // FIXME:
  int __res = vasprintf(__s, __format, __va);
  va_end(__va);
  return __res;
}

int sscanf_l(const char *__s, locale_t __l, const char *__format, ...) {
  va_list __va;
  va_start(__va, __format);
  // FIXME:
  int __res = vsscanf(__s, __format, __va);
  va_end(__va);
  return __res;
}

locale_t newlocale(int mask, const char *locale, locale_t base) {

  if ((locale == NULL) || (locale[0] == '\0') ||
      ((locale[0] == 'C') && (locale[1] == '\0')))
  {
    return __C_locale;
  }

  // Solaris locales are shared libraries that contain 
  char *path;
#ifdef __LP64
  asprintf(&path, "/usr/lib/locale/%1$s/amd64/%1$s.so.3", locale);
#else
  asprintf(&path, "/usr/lib/locale/%1$s/%1$s.so.3", locale);
#endif
  void *handle = dlopen(path, RTLD_LOCAL | RTLD_NOW);
  free(path);
  if (!handle) 
    return 0;
  _LC_locale_t *(*init)() = dlsym(handle, "instantiate");
  if (!init)
    return 0;
  _LC_locale_t  *p = init();
  if (!p)
    return 0;

  if (!base)
    base = __C_locale;

  locale_t ret = calloc(1, sizeof(struct _LC_locale_t));
  memcpy(ret, p, sizeof (_LC_locale_t));
  ret->lc_collate = (mask & LC_COLLATE_MASK) ? p->lc_collate : base->lc_collate;
  ret->lc_ctype = (mask & LC_CTYPE_MASK) ? p->lc_ctype : base->lc_ctype;
  ret->lc_messages = (mask & LC_MESSAGES_MASK) ? p->lc_messages : base->lc_messages;
  ret->lc_monetary = (mask & LC_MONETARY_MASK) ? p->lc_monetary : base->lc_monetary;
  ret->lc_time = (mask & LC_TIME_MASK) ? p->lc_time : base->lc_time;
  return ret;
}

void freelocale(locale_t loc)
{
  if (loc != __C_locale)
    free(loc);
}

__attribute__((constructor))
static void setupCLocale(void) {
  // The default initial locale is the C locale.  This is a statically
  // allocated locale inside libc.  At program start, __lc_locale will point to
  // this.  We need to grab a copy because it's not a public symbol.  If we had
  // access to the source code for libc, then we'd just use it directly...
  assert('C' == setlocale(LC_ALL, 0)[0]);
  __C_locale = __lc_locale;
}
#endif
