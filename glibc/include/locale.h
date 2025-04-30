#ifndef	_LOCALE_H
#include <locale/locale.h>

#ifndef _ISOMAC
extern __typeof (uselocale) __uselocale;

libc_hidden_proto (setlocale)
libc_hidden_proto (__uselocale)

/* This has to be changed whenever a new locale is defined.  */
#define __LC_LAST	13

extern struct loaded_l10nfile *_nl_locale_file_list[] attribute_hidden;

/* Locale object for C locale.  */
extern const struct __locale_struct _nl_C_locobj attribute_hidden;
#define _nl_C_locobj_ptr ((struct __locale_struct *) &_nl_C_locobj)

/* Now define the internal interfaces.  */
extern struct lconv *__localeconv (void);

/* Fetch the name of the current locale set in the given category.  */
extern const char *__current_locale_name (int category) attribute_hidden;

#endif
#endif
