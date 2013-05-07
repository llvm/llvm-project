////////////////////////////////////////////////////////////////////////////////
// Minimal xlocale implementation for Solaris.  This implements the subset of
// the xlocale APIs that libc++ depends on.
////////////////////////////////////////////////////////////////////////////////
#ifndef __XLOCALE_H_INCLUDED
#define __XLOCALE_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif


typedef struct _LC_locale_t* locale_t;

#define LC_COLLATE_MASK  (1<<0)
#define LC_CTYPE_MASK    (1<<1)
#define LC_MESSAGES_MASK (1<<2)
#define LC_MONETARY_MASK (1<<3)
#define LC_NUMERIC_MASK  (1<<4)
#define LC_TIME_MASK     (1<<5)
#define LC_ALL_MASK      (LC_COLLATE_MASK | LC_CTYPE_MASK | LC_MESSAGES_MASK | \
                LC_MONETARY_MASK | LC_NUMERIC_MASK | LC_TIME_MASK)

#define LC_GLOBAL_LOCALE ((locale_t)-1)

size_t __mb_cur_max(locale_t l);
#define MB_CUR_MAX_L(l) __mb_cur_max(l) 

locale_t newlocale(int mask, const char * locale, locale_t base);
void freelocale(locale_t loc);

wint_t btowc_l(int __c, locale_t __l);

int wctob_l(wint_t __c, locale_t __l);

size_t wcrtomb_l(char *__s, wchar_t __wc, mbstate_t *__ps, locale_t __l);

size_t mbrtowc_l(wchar_t *__pwc, const char *__s, size_t __n,
                 mbstate_t *__ps, locale_t __l);

int mbtowc_l(wchar_t *__pwc, const char *__pmb, size_t __max, locale_t __l);

size_t mbrlen_l(const char *__s, size_t __n, mbstate_t *__ps, locale_t __l);

struct lconv *localeconv_l(locale_t __l);

size_t mbsrtowcs_l(wchar_t *__dest, const char **__src, size_t __len,
                   mbstate_t *__ps, locale_t __l);

int sprintf_l(char *__s, locale_t __l, const char *__format, ...);

int snprintf_l(char *__s, size_t __n, locale_t __l, const char *__format, ...);

int asprintf_l(char **__s, locale_t __l, const char *__format, ...);

int sscanf_l(const char *__s, locale_t __l, const char *__format, ...);

int isalnum_l(int,locale_t);
int isalpha_l(int,locale_t);
int isblank_l(int,locale_t);
int iscntrl_l(int,locale_t);
int isdigit_l(int,locale_t);
int isgraph_l(int,locale_t);
int islower_l(int,locale_t);
int isprint_l(int,locale_t);
int ispunct_l(int,locale_t);
int isspace_l(int,locale_t);
int isupper_l(int,locale_t);
int isxdigit_l(int,locale_t);

int iswalnum_l(wchar_t,locale_t);
int iswalpha_l(wchar_t,locale_t);
int iswblank_l(wchar_t,locale_t);
int iswcntrl_l(wchar_t,locale_t);
int iswdigit_l(wchar_t,locale_t);
int iswgraph_l(wchar_t,locale_t);
int iswlower_l(wchar_t,locale_t);
int iswprint_l(wchar_t,locale_t);
int iswpunct_l(wchar_t,locale_t);
int iswspace_l(wchar_t,locale_t);
int iswupper_l(wchar_t,locale_t);
int iswxdigit_l(wchar_t,locale_t);

int iswctype_l(wint_t, wctype_t, locale_t);

int toupper_l(int __c, locale_t __l);
int tolower_l(int __c, locale_t __l);
wint_t towupper_l(wint_t __c, locale_t __l);
wint_t towlower_l(wint_t __c, locale_t __l);


int strcoll_l(const char *__s1, const char *__s2, locale_t __l);
int wcscoll_l(const wchar_t *__s1, const wchar_t *__s2, locale_t __l);
size_t strftime_l(char *__s, size_t __size, const char *__fmt, const struct tm
    *__tm, locale_t __l);

size_t strxfrm_l(char *__s1, const char *__s2, size_t __n, locale_t __l);

size_t wcsxfrm_l(wchar_t *__ws1, const wchar_t *__ws2, size_t __n,
    locale_t __l);



size_t
mbsnrtowcs_l(wchar_t * __restrict dst, const char ** __restrict src,
    size_t nms, size_t len, mbstate_t * __restrict ps, locale_t loc);


size_t
wcsnrtombs_l(char * __restrict dst, const wchar_t ** __restrict src,
    size_t nwc, size_t len, mbstate_t * __restrict ps, locale_t loc);

locale_t __cloc(void);

// FIXME: These are quick-and-dirty hacks to make things pretend to work
static inline
long long strtoll_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoll(__nptr, __endptr, __base);
}
static inline
long strtol_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtol(__nptr, __endptr, __base);
}
static inline
long double strtold_l(const char *__nptr, char **__endptr,
    locale_t __loc) {
  return strtold(__nptr, __endptr);
}
static inline
unsigned long long strtoull_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoull(__nptr, __endptr, __base);
}
static inline
unsigned long strtoul_l(const char *__nptr, char **__endptr,
    int __base, locale_t __loc) {
  return strtoul(__nptr, __endptr, __base);
}

#ifdef __cplusplus
}
#endif

#endif
