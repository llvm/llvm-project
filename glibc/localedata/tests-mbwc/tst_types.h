/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	tst_types.h
 *
 *	 Definitions of data types for each test function
 */


#ifndef TST_TYPES_H
#define TST_TYPES_H

#include <stdio.h>
#include <wchar.h>
#include <wctype.h>
#include "tst_funcs.h"
#include "tgn_funcdef.h"

#define MBSSIZE		   24
#define WCSSIZE		   12
#define MONFMTSIZE	   16
#define MONSIZE		   64
#define USE_MBCURMAX	   99	/* well, but ... */
#define TST_DBL_EPS	   2.22153e-16
#define WCSTOK_SEQNUM	   3
#define MBLEN_SEQNUM	   3
#define MBTOWC_SEQNUM	   3
#define MBSTOWCS_SEQNUM	   3
#define WCTOMB_SEQNUM	   3
#define WCSTOMBS_SEQNUM	   3
#define MBRLEN_SEQNUM	   3
#define MBRTOWC_SEQNUM	   3
#define MBSRTOWCS_SEQNUM   3
#define WCRTOMB_SEQNUM	   3
#define WCSRTOMBS_SEQNUM   3

/* Maximum numbers of test in one of the _loc arrays.  */
#define MAX_LOC_TEST		300


/*----------------------------------------------------------------------*/
/*  FUNCTION								*/
/*----------------------------------------------------------------------*/

typedef struct
{
  char *func_str;
  int func_id;
}
TST_FID;

typedef struct
{
  int func_id;
  const char *locale;
}
TST_HEAD;

typedef struct
{
  TST_HEAD *head;
}
TST_FUNCS;


/*----------------------------------------------------------------------*/
/*  ISW*: int isw* (wchar_t wc)						*/
/*----------------------------------------------------------------------*/

TST_ISW_STRUCT (ALNUM, alnum);
TST_ISW_STRUCT (ALPHA, alpha);
TST_ISW_STRUCT (CNTRL, cntrl);
TST_ISW_STRUCT (DIGIT, digit);
TST_ISW_STRUCT (GRAPH, graph);
TST_ISW_STRUCT (LOWER, lower);
TST_ISW_STRUCT (PRINT, print);
TST_ISW_STRUCT (PUNCT, punct);
TST_ISW_STRUCT (SPACE, space);
TST_ISW_STRUCT (UPPER, upper);
TST_ISW_STRUCT (XDIGIT, xdigit);

typedef struct
{
  wint_t wc;
  const char *ts;
}
TIN_ISWCTYPE_REC;

typedef
TEX_ERRET_REC (int)
  TEX_ISWCTYPE_REC;
TMD_RECHEAD (ISWCTYPE);


/*----------------------------------------------------------------------*/
/*  MBLEN: int mblen (const char *s, size_t n)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  char s_flg;
  char s[MBSSIZE];
  size_t n;
}
TIN_MBLEN_REC;

typedef TEX_ERRET_REC (int) TEX_MBLEN_REC;
TMD_RECHEAD (MBLEN);


/*----------------------------------------------------------------------*/
/*  MBRLEN: size_t mbrlen (const char *s, size_t n, mbstate_t *ps)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  struct
  {
    int s_flg;
    char s[MBSSIZE];
    size_t n;
    int t_flg;
    int t_init;
  }
  seq[MBRLEN_SEQNUM];
}
TIN_MBRLEN_REC;

typedef TEX_ERRET_REC_SEQ (size_t, MBRLEN_SEQNUM) TEX_MBRLEN_REC;
TMD_RECHEAD (MBRLEN);


/*----------------------------------------------------------------------*/
/*  MBRTOWC: size_t mbrtowc (wchar_t *pwc, const char *s, size_t n,	*/
/*			     mbstate_t *ps)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  struct
  {
    int w_flg;
    int s_flg;
    char s[MBSSIZE];
    size_t n;
    int t_flg;
    int t_init;
  }
  seq[MBRTOWC_SEQNUM];
}
TIN_MBRTOWC_REC;

typedef struct
{
  struct
  {
    TMD_ERRET (size_t);
    wchar_t wc;
  }
  seq[MBRTOWC_SEQNUM];
}
TEX_MBRTOWC_REC;

TMD_RECHEAD (MBRTOWC);


/*----------------------------------------------------------------------*/
/*  MBSRTOWCS: size_t mbsrtowcs (wchar_t *ws, const char **s, size_t n, */
/*				 mbstate_t *ps )			*/
/*----------------------------------------------------------------------*/

typedef struct
{
  struct
  {
    int w_flg;
    char s[MBSSIZE];
    size_t n;
    int t_flg;
    int t_init;
  }
  seq[MBSRTOWCS_SEQNUM];
}
TIN_MBSRTOWCS_REC;

typedef struct
{
  struct
  {
    TMD_ERRET (size_t);
    wchar_t ws[WCSSIZE];
  }
  seq[MBSRTOWCS_SEQNUM];
}
TEX_MBSRTOWCS_REC;

TMD_RECHEAD (MBSRTOWCS);


/*----------------------------------------------------------------------*/
/*  MBSTOWCS: size_t mbstowcs (wchar_t *ws, const char *s, size_t n)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  struct
  {
    int w_flg;
    int s_flg;
    const char *s;
    size_t n;
  }
  seq[MBSTOWCS_SEQNUM];
}
TIN_MBSTOWCS_REC;

typedef TEX_MBSRTOWCS_REC TEX_MBSTOWCS_REC;
/* MBSRTOWCS_SEQNUM == MBSTOWCS_SEQNUM */
TMD_RECHEAD (MBSTOWCS);


/*----------------------------------------------------------------------*/
/*  MBTOWC: int mbtowc (wchar_t *wc, const char *s, size_t n)		*/
/*----------------------------------------------------------------------*/

typedef TIN_MBSTOWCS_REC TIN_MBTOWC_REC;
/* MBTOWC_SEQNUM == MBSTOWCS_SEQNUM */

typedef struct
{
  struct
  {
    TMD_ERRET (int);
    wchar_t wc;
  }
  seq[MBTOWC_SEQNUM];
}
TEX_MBTOWC_REC;

TMD_RECHEAD (MBTOWC);


/*----------------------------------------------------------------------*/
/*  STRCOLL: int strcoll (const char *s1, const char *s2)		*/
/*----------------------------------------------------------------------*/

typedef struct
{
  char s1[MBSSIZE];
  char s2[MBSSIZE];
}
TIN_STRCOLL_REC;

typedef TEX_ERRET_REC (int) TEX_STRCOLL_REC;
TMD_RECHEAD (STRCOLL);


/*----------------------------------------------------------------------*/
/*  STRFMON: size_t strfmon (char *buf, size_t nbytes,			*/
/*			     const char *fmt, ... )			*/
/*----------------------------------------------------------------------*/

typedef struct
{
  int nbytes;
  char fmt[MONFMTSIZE];
  double val;
}
TIN_STRFMON_REC;

typedef struct
{
  TMD_ERRET (size_t);
  char mon[MONSIZE];
}
TEX_STRFMON_REC;

TMD_RECHEAD (STRFMON);


/*----------------------------------------------------------------------*/
/*  STRXFRM: size_t strxfrm (char *s1, const char *s2, size_t n)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  char org1[MBSSIZE];
  char org2[MBSSIZE];
  size_t n1, n2;
}
TIN_STRXFRM_REC;

typedef TEX_ERRET_REC (size_t) TEX_STRXFRM_REC;	/* only for org2[] */
TMD_RECHEAD (STRXFRM);


/*----------------------------------------------------------------------*/
/*  SWSCANF: int swscanf (const wchar_t *ws, const wchar_t *fmt, ...)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws[WCSSIZE * 3];
  wchar_t fmt[WCSSIZE * 3];
  int wch;
}
TIN_SWSCANF_REC;

typedef struct
{
  TMD_ERRET (int);
  int val_int;		/* %d */
  unsigned val_uns;	/* %u */
  float val_flt;		/* %f */
  int val_c;		/* %c */
  char val_s[MBSSIZE * 2];	/* %s */
  wchar_t val_S[WCSSIZE * 2];	/* %lc, %ls, %C, %S */
}
TEX_SWSCANF_REC;

TMD_RECHEAD (SWSCANF);


/*----------------------------------------------------------------------*/
/*  TOWCTRANS: wint_t towctrans (wint_t wc, wctrans_t desc)		*/
/*----------------------------------------------------------------------*/

typedef TIN_ISWCTYPE_REC TIN_TOWCTRANS_REC;
typedef TEX_ERRET_REC (wint_t) TEX_TOWCTRANS_REC;
TMD_RECHEAD (TOWCTRANS);


/*----------------------------------------------------------------------*/
/*  TOW*ER: wint_t tow*er (wint_t wc)					*/
/*----------------------------------------------------------------------*/

TST_TOW_STRUCT (LOWER, lower);
TST_TOW_STRUCT (UPPER, upper);


/*----------------------------------------------------------------------*/
/*  WCRTOMB: wchar_t wcrtomb (char *s, wchar_t wc, mbstate_t *ps)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  int s_flg;
  wchar_t wc;
  int t_flg;
  int t_init;
}
TIN_WCRTOMB_REC;

typedef struct
{
  TMD_ERRET (wchar_t);
  char s[MBSSIZE];
}
TEX_WCRTOMB_REC;

TMD_RECHEAD (WCRTOMB);


/*----------------------------------------------------------------------*/
/*  WCSCAT: wchar_t *wcscat (wchar_t *ws1, wchar_t *ws2)		*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws1[WCSSIZE];
  wchar_t ws2[WCSSIZE];
}
TIN_WCSCAT_REC;

typedef struct
{
  TMD_ERRET (wchar_t *);
  wchar_t ws[WCSSIZE];
}
TEX_WCSCAT_REC;

TMD_RECHEAD (WCSCAT);


/*----------------------------------------------------------------------*/
/*  WCSCHR: wchar_t *wcschr (wchar_t *ws, wchar_t wc);			*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws[WCSSIZE];
  wchar_t wc;
}
TIN_WCSCHR_REC;

typedef TEX_ERRET_REC (wchar_t *) TEX_WCSCHR_REC;
TMD_RECHEAD (WCSCHR);


/*----------------------------------------------------------------------*/
/*  WCSCMP: int wcscmp (const wchar_t *ws1, const wchar_t *ws2)		*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCAT_REC TIN_WCSCMP_REC;
typedef TEX_ERRET_REC (int) TEX_WCSCMP_REC;
TMD_RECHEAD (WCSCMP);


/*----------------------------------------------------------------------*/
/*  WCSCOLL: int wcscoll (const wchar_t *ws1, const wchar_t *ws2)	*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCMP_REC TIN_WCSCOLL_REC;
typedef struct
{
  TMD_ERRET (int);
  int cmp_flg;
}
TEX_WCSCOLL_REC;
TMD_RECHEAD (WCSCOLL);


/*----------------------------------------------------------------------*/
/*  WCSCPY: wchar_t *wcscpy (wchar_t *ws1, const wchar_t *ws2)		*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws[WCSSIZE];	/* ws2: original string */
}
TIN_WCSCPY_REC;

typedef TEX_WCSCAT_REC TEX_WCSCPY_REC;
TMD_RECHEAD (WCSCPY);


/*----------------------------------------------------------------------*/
/*  WCSCSPN: size_t wcscspn (const wchar_t *ws1, const wchar_t *ws2)	*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCAT_REC TIN_WCSCSPN_REC;
typedef TEX_ERRET_REC (size_t) TEX_WCSCSPN_REC;
TMD_RECHEAD (WCSCSPN);


/*----------------------------------------------------------------------*/
/*  WCSLEN: size_t wcslen (const wchar_t *ws)				*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCPY_REC TIN_WCSLEN_REC;
typedef TEX_ERRET_REC (size_t) TEX_WCSLEN_REC;
TMD_RECHEAD (WCSLEN);


/*----------------------------------------------------------------------*/
/*  WCSNCAT: wchar_t *wcsncat (wchar_t *ws1, const wchar_t *ws2,	*/
/*			       size_t n)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws1[WCSSIZE];
  wchar_t ws2[WCSSIZE];
  size_t n;
}
TIN_WCSNCAT_REC;

typedef TEX_WCSCAT_REC TEX_WCSNCAT_REC;
TMD_RECHEAD (WCSNCAT);


/*----------------------------------------------------------------------*/
/*  WCSNCMP: int *wcsncmp (const wchar_t *ws1, const wchar_t *ws2,	*/
/*			   size_t n)					*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSNCAT_REC TIN_WCSNCMP_REC;
typedef TEX_ERRET_REC (int) TEX_WCSNCMP_REC;
TMD_RECHEAD (WCSNCMP);


/*----------------------------------------------------------------------*/
/*  WCSNCPY: wchar_t *wcsncpy (wchar_t *ws1, const wchar_t *ws2,	*/
/*			       size_t n)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t ws[WCSSIZE];	/* ws2: original string */
  size_t n;
}
TIN_WCSNCPY_REC;

typedef TEX_WCSCPY_REC TEX_WCSNCPY_REC;
TMD_RECHEAD (WCSNCPY);


/*----------------------------------------------------------------------*/
/*  WCSPBRK: wchar_t *wcspbrk (const wchar_t *ws1, const wchar_t *ws2)	*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCSPN_REC TIN_WCSPBRK_REC;

typedef struct
{
  TMD_ERRET (wchar_t *);
  wchar_t wc;
}
TEX_WCSPBRK_REC;

TMD_RECHEAD (WCSPBRK);


/*----------------------------------------------------------------------*/
/*  WCSRTOMBS: size_t wcsrtombs (char *s, const wchar_t **ws, size_t n, */
/*				 mbstate_t *ps)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  int s_flg;
  int w_flg;		/* don't need this */
  wchar_t ws[WCSSIZE];
  size_t n;
  int t_flg;
  int t_init;
}
TIN_WCSRTOMBS_REC;

typedef struct
{
  TMD_ERRET (size_t);
  char s[MBSSIZE];
}
TEX_WCSRTOMBS_REC;

TMD_RECHEAD (WCSRTOMBS);


/*----------------------------------------------------------------------*/
/*  WCSSPN: size_t wcsspn (const wchar_t *ws1, const wchar_t *ws2)	*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCSPN_REC TIN_WCSSPN_REC;
typedef TEX_WCSCSPN_REC TEX_WCSSPN_REC;
TMD_RECHEAD (WCSSPN);


/*----------------------------------------------------------------------*/
/*  WCSSTR: wchar_t *wcsstr (const wchar_t *ws1, const wchar_t *ws2)	*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSCSPN_REC TIN_WCSSTR_REC;
typedef TEX_ERRET_REC (wchar_t *) TEX_WCSSTR_REC;
TMD_RECHEAD (WCSSTR);


/*----------------------------------------------------------------------*/
/*  WCSTOD: double wcstod (const wchar_t *np, wchar_t **endp)		*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t np[WCSSIZE];
}
TIN_WCSTOD_REC;

typedef struct
{
  TMD_ERRET (double);
  double val;
  wchar_t fwc;
}
TEX_WCSTOD_REC;

TMD_RECHEAD (WCSTOD);


/*----------------------------------------------------------------------*/
/*  WCSTOK: wchar_t *wcstok (wchar_t *ws, const wchar_t *dlm,		*/
/*			     wchar_t **pt)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  struct
  {
    int w_flg;
    wchar_t ws[WCSSIZE];
    wchar_t dt[WCSSIZE];	/* delimiter */
  }
  seq[WCSTOK_SEQNUM];
}
TIN_WCSTOK_REC;

typedef struct
{
  struct
  {
    TMD_ERRET (wchar_t *);
    wchar_t ws[WCSSIZE];
  }
  seq[WCSTOK_SEQNUM];
}
TEX_WCSTOK_REC;

TMD_RECHEAD (WCSTOK);


/*----------------------------------------------------------------------*/
/*  WCSTOMBS: size_t wcstombs (char s, const wchar_t *ws, size_t n)	*/
/*----------------------------------------------------------------------*/

typedef struct
{
  int s_flg;
  int w_flg;		/* currently we don't need it. */
  wchar_t ws[WCSSIZE];
  size_t n;
}
TIN_WCSTOMBS_REC;

typedef struct
{
  TMD_ERRET (size_t);
  char s[MBSSIZE];
}
TEX_WCSTOMBS_REC;

TMD_RECHEAD (WCSTOMBS);


/*----------------------------------------------------------------------*/
/*  WCSWIDTH: int wcswidth (const wchar_t *ws, size_t n)		*/
/*----------------------------------------------------------------------*/

typedef TIN_WCSNCPY_REC TIN_WCSWIDTH_REC;
typedef TEX_ERRET_REC (int) TEX_WCSWIDTH_REC;
TMD_RECHEAD (WCSWIDTH);


/*----------------------------------------------------------------------*/
/*  WCSXFRM: size_t wcsxfrm (wchar_t *ws1, const wchar_t *ws2, size_t n)*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t org1[WCSSIZE];
  wchar_t org2[WCSSIZE];
  int n1, n2;
}
TIN_WCSXFRM_REC;

typedef TEX_ERRET_REC (size_t) TEX_WCSXFRM_REC;	/* only for org2[] */
TMD_RECHEAD (WCSXFRM);


/*----------------------------------------------------------------------*/
/*  WCTOB: int wctob (wint_t wc)					*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wint_t wc;
}
TIN_WCTOB_REC;

typedef TEX_ERRET_REC (int) TEX_WCTOB_REC;
TMD_RECHEAD (WCTOB);


/*----------------------------------------------------------------------*/
/*  WCTOMB: int wctomb (char *s, wchar_t wc)				*/
/*----------------------------------------------------------------------*/

typedef struct
{
  int s_flg;
  wchar_t wc;
}
TIN_WCTOMB_REC;

typedef struct
{
  TMD_ERRET (int);
  char s[MBSSIZE];
}
TEX_WCTOMB_REC;

TMD_RECHEAD (WCTOMB);


/*----------------------------------------------------------------------*/
/*  WCTRANS: wctrans_t wctrans (const char *charclass)			*/
/*----------------------------------------------------------------------*/

typedef struct
{
  char class[MBSSIZE];
}
TIN_WCTRANS_REC;

typedef TEX_ERRET_REC (wctrans_t) TEX_WCTRANS_REC;
TMD_RECHEAD (WCTRANS);


/*----------------------------------------------------------------------*/
/*  WCTYPE: wctype_t wctype (const char *class)				*/
/*----------------------------------------------------------------------*/

typedef TIN_WCTRANS_REC TIN_WCTYPE_REC;
typedef TEX_ERRET_REC (wctype_t) TEX_WCTYPE_REC;
TMD_RECHEAD (WCTYPE);


/*----------------------------------------------------------------------*/
/*  WCWIDTH: int wcwidth (wchar_t wc)					*/
/*----------------------------------------------------------------------*/

typedef struct
{
  wchar_t wc;
}
TIN_WCWIDTH_REC;

typedef TEX_ERRET_REC (int) TEX_WCWIDTH_REC;
TMD_RECHEAD (WCWIDTH);

#endif /* TST_TYPES_H */
