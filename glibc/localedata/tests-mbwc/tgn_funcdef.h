#ifndef TGN_FUNCDEF_H
#define TGN_FUNCDEF_H

/* Unique number for each test.  */
#define	 Tiswalnum		1
#define	 Tiswalpha		2
#define	 Tiswcntrl		3
#define	 Tiswctype		4
#define	 Tiswdigit		5
#define	 Tiswgraph		6
#define	 Tiswlower		7
#define	 Tiswprint		8
#define	 Tiswpunct		9
#define	 Tiswspace		10
#define	 Tiswupper		11
#define	 Tiswxdigit		12
#define	 Tmblen			13
#define	 Tmbrlen		14
#define	 Tmbrtowc		15
#define	 Tmbsrtowcs		16
#define	 Tmbstowcs		17
#define	 Tmbtowc		18
#define	 Tstrcoll		19
#define	 Tstrfmon		20
#define	 Tstrxfrm		21
#define	 Tswscanf		22
#define	 Ttowctrans		23
#define	 Ttowlower		24
#define	 Ttowupper		25
#define	 Twcrtomb		26
#define	 Twcscat		27
#define	 Twcschr		28
#define	 Twcscmp		29
#define	 Twcscoll		30
#define	 Twcscpy		31
#define	 Twcscspn		32
#define	 Twcslen		33
#define	 Twcsncat		34
#define	 Twcsncmp		35
#define	 Twcsncpy		36
#define	 Twcspbrk		37
#define	 Twcsrtombs		38
#define	 Twcsspn		39
#define	 Twcsstr		40
#define	 Twcstod		41
#define	 Twcstok		42
#define	 Twcstombs		43
#define	 Twcswidth		44
#define	 Twcsxfrm		45
#define	 Twctob			46
#define	 Twctomb		47
#define	 Twctrans		48
#define	 Twctype		49
#define	 Twcwidth		50

/* Name of each test.  */
#define	 S_ISWALNUM	"iswalnum"
#define	 S_ISWALPHA	"iswalpha"
#define	 S_ISWCNTRL	"iswcntrl"
#define	 S_ISWCTYPE	"iswctype"
#define	 S_ISWDIGIT	"iswdigit"
#define	 S_ISWGRAPH	"iswgraph"
#define	 S_ISWLOWER	"iswlower"
#define	 S_ISWPRINT	"iswprint"
#define	 S_ISWPUNCT	"iswpunct"
#define	 S_ISWSPACE	"iswspace"
#define	 S_ISWUPPER	"iswupper"
#define	 S_ISWXDIGIT	"iswxdigit"
#define	 S_MBLEN	"mblen"
#define	 S_MBRLEN	"mbrlen"
#define	 S_MBRTOWC	"mbrtowc"
#define	 S_MBSRTOWCS	"mbsrtowcs"
#define	 S_MBSTOWCS	"mbstowcs"
#define	 S_MBTOWC	"mbtowc"
#define	 S_STRCOLL	"strcoll"
#define	 S_STRFMON	"strfmon"
#define	 S_STRXFRM	"strxfrm"
#define	 S_SWSCANF	"swscanf"
#define	 S_TOWCTRANS	"towctrans"
#define	 S_TOWLOWER	"towlower"
#define	 S_TOWUPPER	"towupper"
#define	 S_WCRTOMB	"wcrtomb"
#define	 S_WCSCAT	"wcscat"
#define	 S_WCSCHR	"wcschr"
#define	 S_WCSCMP	"wcscmp"
#define	 S_WCSCOLL	"wcscoll"
#define	 S_WCSCPY	"wcscpy"
#define	 S_WCSCSPN	"wcscspn"
#define	 S_WCSLEN	"wcslen"
#define	 S_WCSNCAT	"wcsncat"
#define	 S_WCSNCMP	"wcsncmp"
#define	 S_WCSNCPY	"wcsncpy"
#define	 S_WCSPBRK	"wcspbrk"
#define	 S_WCSRTOMBS	"wcsrtombs"
#define	 S_WCSSPN	"wcsspn"
#define	 S_WCSSTR	"wcsstr"
#define	 S_WCSTOD	"wcstod"
#define	 S_WCSTOK	"wcstok"
#define	 S_WCSTOMBS	"wcstombs"
#define	 S_WCSWIDTH	"wcswidth"
#define	 S_WCSXFRM	"wcsxfrm"
#define	 S_WCTOB	"wctob"
#define	 S_WCTOMB	"wctomb"
#define	 S_WCTRANS	"wctrans"
#define	 S_WCTYPE	"wctype"
#define	 S_WCWIDTH	"wcwidth"

/* Prototypes for test functions.  */
extern int tst_iswalnum (FILE *, int);
extern int tst_iswalpha (FILE *, int);
extern int tst_iswcntrl (FILE *, int);
extern int tst_iswctype (FILE *, int);
extern int tst_iswdigit (FILE *, int);
extern int tst_iswgraph (FILE *, int);
extern int tst_iswlower (FILE *, int);
extern int tst_iswprint (FILE *, int);
extern int tst_iswpunct (FILE *, int);
extern int tst_iswspace (FILE *, int);
extern int tst_iswupper (FILE *, int);
extern int tst_iswxdigit (FILE *, int);
extern int tst_mblen (FILE *, int);
extern int tst_mbrlen (FILE *, int);
extern int tst_mbrtowc (FILE *, int);
extern int tst_mbsrtowcs (FILE *, int);
extern int tst_mbstowcs (FILE *, int);
extern int tst_mbtowc (FILE *, int);
extern int tst_strcoll (FILE *, int);
extern int tst_strfmon (FILE *, int);
extern int tst_strxfrm (FILE *, int);
extern int tst_swscanf (FILE *, int);
extern int tst_towctrans (FILE *, int);
extern int tst_towlower (FILE *, int);
extern int tst_towupper (FILE *, int);
extern int tst_wcrtomb (FILE *, int);
extern int tst_wcscat (FILE *, int);
extern int tst_wcschr (FILE *, int);
extern int tst_wcscmp (FILE *, int);
extern int tst_wcscoll (FILE *, int);
extern int tst_wcscpy (FILE *, int);
extern int tst_wcscspn (FILE *, int);
extern int tst_wcslen (FILE *, int);
extern int tst_wcsncat (FILE *, int);
extern int tst_wcsncmp (FILE *, int);
extern int tst_wcsncpy (FILE *, int);
extern int tst_wcspbrk (FILE *, int);
extern int tst_wcsrtombs (FILE *, int);
extern int tst_wcsspn (FILE *, int);
extern int tst_wcsstr (FILE *, int);
extern int tst_wcstod (FILE *, int);
extern int tst_wcstok (FILE *, int);
extern int tst_wcstombs (FILE *, int);
extern int tst_wcswidth (FILE *, int);
extern int tst_wcsxfrm (FILE *, int);
extern int tst_wctob (FILE *, int);
extern int tst_wctomb (FILE *, int);
extern int tst_wctrans (FILE *, int);
extern int tst_wctype (FILE *, int);
extern int tst_wcwidth (FILE *, int);

#endif /* TGN_FUNCDEF_H */
